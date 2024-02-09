# Copyright 2020-2024 The FCEst Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from gpflow import likelihoods
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class WishartProcessLikelihood(likelihoods.MonteCarloLikelihood):
    """
    Class for Wishart process likelihoods.
    """

    def __init__(
            self,
            D: int,
            nu: int,
            n_mc_samples: int = 2,
            A_scale_matrix_option: str = 'train_full_matrix',
            train_additive_noise: bool = False,
            additive_noise_matrix_init: float = 0.01,
            verbose: bool = True,
    ) -> None:
        """
        Initialize the Wishart process likelihood.

        Parameters
        ----------
        :param D: 
            The number of time series.
        :param nu: 
            Degrees of freedom.
        :param n_mc_samples:
            Number of Monte Carlo samples used to approximate gradients (S).
        :param A_scale_matrix_option:
        :param train_additive_noise:
            Whether to train the additive noise matrix (Lambda).
        :param additive_noise_matrix_init:
            Initial value of additive noise.
            Note: Heaukulani2019 used 0.001.
            However, empirical results seem to indicate that 0.01 is much better than 0.001.
        :param verbose:
        """
        nu = D if nu is None else nu
        if nu < D:
            raise Exception("Wishart Degrees of Freedom must be >= D.")
        super().__init__(
            latent_dim=D*nu,
            observation_dim=D
        )
        self.D = D
        self.nu = nu
        self.n_mc_samples = n_mc_samples
        self.A_scale_matrix = self._set_A_scale_matrix(option=A_scale_matrix_option)  # (D, D)

        # The additive noise matrix must have positive diagonal values, which this softplus construction guarantees.
        additive_noise_matrix_init = np.log(
            np.exp(additive_noise_matrix_init) - 1
        )  # inverse softplus
        self.additive_part = tf.Variable(
            additive_noise_matrix_init * tf.ones(D, dtype=tf.float64),
            dtype=tf.float64,
            trainable=train_additive_noise
        )  # (D, )

        if verbose:
            logging.info(f"A scale matrix option is '{A_scale_matrix_option:s}'.")
            print('A_scale_matrix: ', self.A_scale_matrix)
            print('initial additive part: ', self.additive_part)

    def variational_expectations(
            self, f_mean: tf.Tensor, f_variance: tf.Tensor, y_data: np.array
    ) -> tf.Tensor:
        """
        This returns the expectation of log likelihood part of the ELBO.
        That is, it computes log p(Y | variational parameters).
        This does not include the KL term.
        Models inheriting VGP are required to have this signature.

        Parameters
        ----------
        :param f_mean:
            (N, D * nu), the parameters of the latent GP points F
        :param f_variance:
            (N, D * nu), the parameters of the latent GP points F
        :param y_data:
            NumPy array of shape (n_time_steps, n_time_series) or (N, D).
        :return:
            Tensor of shape (N, ); logp, log probability density of the data.
        """
        N, _ = y_data.shape
        f_stddev = f_variance ** 0.5

        # produce (multiple) samples of F, the matrix with latent GP points at input locations X
        f_sample = tf.random.normal(
            (self.n_mc_samples, N, self.D * self.nu),
            mean=0.0, stddev=1.0,
            dtype=tf.dtypes.float64
        ) * f_stddev + f_mean  # (S, N, D * nu)
        f_sample = tf.reshape(f_sample, (self.n_mc_samples, N, self.D, -1))  # (S, N, D, nu)

        # finally, the likelihood variant will use these to compute the appropriate log density
        return self._log_prob(f_sample, y_data)  # (N, )

    def _log_prob(self, f_sample: tf.Tensor, y_data: np.array) -> tf.Tensor:
        """
        Compute the (Monte Carlo estimate of) the log likelihood given samples of the GPs.

        Parameters
        ----------
        :param f_sample:
            (n_mc_samples, n_time_steps, n_time_series, degrees_of_freedom) or (S, N, D, nu) -
        :param y_data:
            (n_time_steps, n_time_series) or (N, D) -
        :return:
            (n_time_steps, ) or (N, )
        """
        # compute the constant term of the log likelihood
        constant_term = - self.D / 2 * tf.math.log(2 * tf.constant(np.pi, dtype=tf.float64))

        # compute the `log(det(AFFA))` component of the log likelihood
        # TODO: this does not work for nu != D
        # af = tf.matmul(self.A_scale_matrix, f_sample)  # (S, N, D, nu)
        af = tf.multiply(self.A_scale_matrix, f_sample)

        affa = tf.matmul(af, af, transpose_b=True)  # (S, N, D, D) - our construction of \Sigma
        affa = self._add_diagonal_additive_noise(affa)  # (S, N, D, D)
        # Before, the trainable additive noise sometimes broke the Cholesky decomposition.
        # This did not happen again after forcing it to be positive.
        # TODO: Can adding positive values to the diagonal ever make a PSD matrix become non-PSD?
        try:
            L = tf.linalg.cholesky(affa)  # (S, N, D, D)
        except InvalidArgumentError as e:
            print(affa)
            print(self.additive_part)
            print(e)
        log_det_affa = 2 * tf.math.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L)),
            axis=2
        )  # (S, N)

        # compute the `Y * inv(cov) * Y` component of the log likelihood
        # we avoid computing the matrix inverse for computational stability
        # the tiling below is inefficient, but can't get the shapes to play well with cholesky_solve otherwise
        n_samples = tf.shape(f_sample)[0]  # could be 1 when computing MAP test metric
        y_data = tf.tile(y_data[None, :, :, None], [n_samples, 1, 1, 1])
        L_solve_y = tf.linalg.triangular_solve(L, y_data, lower=True)  # (S, N, D, 1)
        yaffay = tf.reduce_sum(L_solve_y ** 2.0, axis=(2, 3))  # (S, N)

        # compute the full log likelihood
        log_likel_p = constant_term - 0.5 * log_det_affa - 0.5 * yaffay  # (S, N)
        log_likel_p = tf.math.reduce_mean(log_likel_p, axis=0)  # mean over Monte Carlo samples, (N, )
        return log_likel_p

    def _set_A_scale_matrix(self, option: str = 'identity') -> tf.Tensor:
        """
        A (the Cholesky factor of scale matrix V) represents the mean of estimates.

        Parameters
        ----------
        :param option:
            1) 'identity': we don't train it, and fix it as an identity matrix.
                In the SWPR code, this option was implemented.
            2) 'scaled': identity matrix scaled by nu
            3) 'train_diagonal': we only train the diagonal values and fix the rest as zeros.
                This option was implemented in the Heaukulani2019 paper.
            4) 'train_full_matrix': we train the whole matrix.
        :return:
            Matrix of shape (D, D).
        """
        if option == 'identity':
            A_scale_matrix = tf.Variable(
                tf.ones(self.D, dtype=tf.float64),
                dtype=tf.float64,
                trainable=False
            )  # (D, )
        elif option == 'scaled':
            A_scale_matrix = tf.Variable(
                tf.ones(self.D, dtype=tf.float64) / self.nu,
                dtype=tf.float64,
                trainable=False
            )  # (D, )
        elif option == 'train_diagonal':
            A_scale_matrix = tf.Variable(
                tf.ones(self.D, dtype=tf.float64),
                dtype=tf.float64,
                trainable=True
            )  # (D, )
            # A_scale_matrix = tf.linalg.diag(self.A_diagonal_terms)
        elif option == 'train_full_matrix':
            A_scale_matrix = tf.Variable(
                tf.eye(self.D, dtype=tf.float64),
                dtype=tf.float64,
                trainable=True
            )  # (D, D)
        else:
            raise NotImplementedError(f"Option '{option:s}' for 'A_scale_matrix' not recognized.")
        return A_scale_matrix

    def _add_diagonal_additive_noise(self, cov_matrix):
        """
        Add some noise, either fixed or trained.

        Parameters
        ----------
        :param cov_matrix:
        :return:
        """
        return tf.linalg.set_diag(
            cov_matrix, tf.linalg.diag_part(cov_matrix) + tf.math.softplus(self.additive_part)
        )


class FactoredWishartProcessLikelihood(likelihoods.MonteCarloLikelihood):

    def __init__(self):
        raise NotImplementedError("Factorized Wishart process not implemented yet.")
