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

import json
import logging
import os

import gpflow.kernels
from gpflow.kernels import Kernel
from gpflow import models
import numpy as np
import tensorflow as tf

from ..helpers.array_operations import are_all_positive_definite, convert_tensor_to_correlation
from .likelihoods import WishartProcessLikelihood, FactoredWishartProcessLikelihood

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class VariationalWishartProcess(models.vgp.VGP):
    """
    Base class of the variational Wishart process (VWP) model.
    Most of the work will be done by `gpflow.models.vgp.VGP`.

    TODO: add option for minibatch training
    TODO: shall we convert all to float32 instead of float64 to speed up computation?
        from gpflow.config import default_float
        from gpflow.utilities import to_default_float
        gpflow.config.set_default_float(np.float64)
    TODO: how can we specify different kernel params for different underlying GPs?
        maybe something like k = Matern52(active_dims=[0]) + Matern52(active_dims=[1])?
        https://gpflow.github.io/GPflow/2.5.2/notebooks/advanced/kernels.html
    """

    def __init__(
            self,
            x_observed: np.array,
            y_observed: np.array,
            nu: int = None,
            kernel: Kernel = None,
            num_mc_samples: int = 5,
            A_scale_matrix_option: str = 'train_full_matrix',
            train_additive_noise: bool = True,
            kernel_lengthscale_init: float = 0.3,
            q_sqrt_init: float = 0.001,
            num_factors: int = None,
    ) -> None:
        """
        Initialize Variational Wishart Process (VWP) model.

        Parameters
        ----------
        :param x_observed:
            Expected in shape (num_time_steps, 1), i.e. (N, 1).
        :param y_observed:
            Expected in shape (num_time_steps, num_time_series), i.e. (N, D).
            Expected to have (approximately) mean zero.
        :param nu:
            Degrees of freedom.
            Empirical results suggest setting nu = D is optimal.
        :param kernel:
            GPflow kernel.
        :param num_mc_samples:
            The number of Monte Carlo samples used to approximate the ELBO.
            In the paper this is R, in the code sometimes S.
        :param A_scale_matrix_option:
            We found that training the full matrix yields the best results.
        :param train_additive_noise:
        :param kernel_lengthscale_init:
        :param q_sqrt_init:
            Empirical results suggest a value of 0.001 is slightly better than 0.01.
        :param num_factors:
        """
        self.D = y_observed.shape[1]
        logging.info(f"Found {self.D:d} time series (D = {self.D:d}).")

        if num_factors is not None:
            raise NotImplementedError("Factorized Wishart process not implemented yet.")

        if nu is None:
            nu = self.D
        self.nu = nu

        if kernel is None:
            kernel = gpflow.kernels.Matern52()

        likel = WishartProcessLikelihood(
            D=self.D,
            nu=nu,
            num_mc_samples=num_mc_samples,
            A_scale_matrix_option=A_scale_matrix_option,
            train_additive_noise=train_additive_noise,
        )
        super().__init__(
            data=(x_observed, y_observed),
            kernel=kernel,
            likelihood=likel,
            num_latent_gps=likel.latent_dim,  # number of outputs (multi−output GP)
        )
        self._initialize_parameters(
            kernel_lengthscale_init=kernel_lengthscale_init,
            q_sqrt_init=q_sqrt_init
        )

    def predict_cov(
            self, x_new: np.array, num_mc_samples: int = 300
    ) -> (tf.Tensor, tf.Tensor):
        """
        The main attribute to predict covariance matrices at any point in time.

        Parameters
        ----------
        :param x_new:
        :param num_mc_samples:
            Note: Heaukulani2019 used 300 MC samples for prediction.
        :return:
        """
        cov_samples = self._get_cov_samples(x_new, num_mc_samples)  # (S_new, N_new, D, D)

        cov_mean = tf.math.reduce_mean(cov_samples, axis=0)  # (N_new, D, D)
        assert are_all_positive_definite(cov_mean)
        cov_stddev = tf.math.reduce_std(cov_samples, axis=0)  # (N_new, D, D)

        return cov_mean, cov_stddev

    def predict_corr(
            self, x_new: np.array, num_mc_samples: int = 300
    ) -> (tf.Tensor, tf.Tensor):
        """
        The main attribute to predict correlation matrices at any point in time.

        TODO: how should we convert sampling uncertainty to correlation confidence interval?

        Parameters
        ----------
        :param x_new:
        :param num_mc_samples:
        :return:
            Tuple of (mean, stddev) of correlation matrices.
        """
        cov_samples = self._get_cov_samples(x_new, num_mc_samples)  # (S_new, N_new, D, D)

        corr_samples = convert_tensor_to_correlation(cov_samples)  # (S_new, N_new, D, D)

        corr_mean = tf.math.reduce_mean(corr_samples, axis=0)  # (N_new, D, D)
        corr_stddev = tf.math.reduce_std(corr_samples, axis=0)  # (N_new, D, D)

        return corr_mean, corr_stddev

    def _get_cov_samples(
            self, x_new: np.array, num_mc_samples: int = 300
    ) -> tf.Tensor:
        """
        Prediction routine for covariance matrices.
        We could switch to a MLE routine for this too, i.e. removing the dependency on Monte Carlo samples.
        But in the Bayesian setting the sampling should work well.

        TODO: should we add the additive noise at prediction time?

        Parameters
        ----------
        :param x_new:
            The (new) locations to get covariance matrix for.
            Can be different from those in training data.
        :param num_mc_samples:
            S_new: number of Monte Carlo samples to approximate covariance matrix with.
            300 samples are used in previous paper.
        :return:
            AFFA here are the constructed covariance matrix samples.
        """
        num_test_time_steps = x_new.shape[0]

        # TODO: maybe we can get f_samples instead of f?
        f_mean_new, f_variance_new = self.predict_f(x_new)  # (N_new, D * nu), (N_new, D * nu)
        f_mean_new = tf.reshape(f_mean_new, [num_test_time_steps, self.D, -1])  # (N_new, D, nu)
        f_variance_new = tf.reshape(f_variance_new, [num_test_time_steps, self.D, -1])  # (N_new, D, nu)
        f_stddev_new = f_variance_new ** 0.5  # (N_new, D, nu)

        f_sample = tf.random.normal(
            (num_mc_samples, num_test_time_steps, self.D, self.nu),
            mean=0.0, stddev=1.0,
            dtype=tf.dtypes.float64
        ) * f_stddev_new + f_mean_new  # (S_new, N_new, D, nu)

        # TODO: this does not work for nu != D
        # af = tf.matmul(self.likelihood.A_scale_matrix, f_sample)  # (S_new, N_new, D, nu)
        af = tf.multiply(self.likelihood.A_scale_matrix, f_sample)
        affa = tf.matmul(af, af, transpose_b=True)  # (S_new, N_new, D, D)
        affa = self.likelihood._add_diagonal_additive_noise(affa)  # (S_new, N_new, D, D)

        return affa

    def _initialize_parameters(
            self, kernel_lengthscale_init: float, q_sqrt_init: float
    ) -> None:
        """
        Model parameter initialization is crucial.
        self.kernel.lengthscales here is a gpflow.base.Parameter object.

        Parameters
        ----------
        :param kernel_lengthscale_init:
            Float value to initialize all kernel lengthscales with.
        :param q_sqrt_init:
        """
        try:
            # Initialize kernel lengthscales as vector if it is specified as such.
            kernel_lengthscale_init = np.ones_like(self.kernel.lengthscales) * kernel_lengthscale_init

            self.kernel.lengthscales.assign(kernel_lengthscale_init)
        except:
            print('Non-standard kernel detected.')
        # m.q_mu.assign(m.q_mu + 0.5)
        self.q_sqrt.assign(self.q_sqrt * q_sqrt_init)

    def save_model_params_dict(
            self, savedir: str, model_name: str
    ) -> None:
        """
        We only save the trained model parameters.
        At loading time, we re-instantiate the model and assign the saved model parameters.
        This currently only works for our Matern52 kernel!

        Parameters
        ----------
        :param savedir:
        :param model_name:
            A string that ends in .json
        """
        params_dict = {
            'D': self.D,
            'nu': self.nu,
            'A_scale_matrix': self.likelihood.A_scale_matrix.numpy().tolist(),
            'additive_noise': self.likelihood.additive_part.numpy().tolist(),
            'kernel_variance': float(self.kernel.variance.numpy()),
            'kernel_lengthscales': float(self.kernel.lengthscales.numpy()),
            'q_mu': self.q_mu.numpy().tolist(),
            'q_sqrt': self.q_sqrt.numpy().tolist()
        }
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir, model_name), 'w') as fp:
            json.dump(params_dict, fp)
        logging.info(f"Model '{model_name:s}' saved in '{savedir:s}'.")

    def load_from_params_dict(
            self, savedir: str, model_name: str
    ) -> None:
        """
        This assumes you have created a new model.
        All trained model parameters are assigned.

        Parameters
        ----------
        :param savedir:
        :param model_name:
        """
        with open(os.path.join(savedir, model_name), 'r') as fp:
            params_dict = json.load(fp)

        A_scale_matrix = np.array(params_dict['A_scale_matrix'])
        self.likelihood.A_scale_matrix.assign(A_scale_matrix)

        additive_part = np.array(params_dict['additive_noise'])
        self.likelihood.additive_part.assign(additive_part)

        kernel_variance = params_dict['kernel_variance']
        self.kernel.variance.assign(kernel_variance)

        kernel_lengthscales = params_dict['kernel_lengthscales']
        self.kernel.lengthscales.assign(kernel_lengthscales)

        q_mu = np.array(params_dict['q_mu'])
        self.q_mu.assign(q_mu)

        q_sqrt = np.array(params_dict['q_sqrt'])
        self.q_sqrt.assign(q_sqrt)

        logging.info(f"VWP model loaded from '{savedir:s}'.")


class SparseVariationalWishartProcess(models.svgp.SVGP):
    """
    Base class of the sparse variational Wishart process (SVWP) model.
    Most of the work will be done by `gpflow.models.svgp.SVGP`.
    This sparse implementation reduces computational cost if we have large N, but is not likely to improve performance.
    However, the location of the inducing points Z may be interesting by themselves.

    TODO: there is still some overlap with the non-sparse VariationalWishartProcess: we could merge some stuff
    """

    def __init__(
            self,
            D: int,
            Z: np.array,
            nu: int = None,
            kernel: Kernel = gpflow.kernels.Matern52(),
            num_mc_samples: int = 5,
            A_scale_matrix_option: str = 'train_full_matrix',
            train_additive_noise: bool = True,
            kernel_lengthscale_init: float = 0.3,
            q_sqrt_init: float = 0.001,
            num_factors: int = None,
            verbose: bool = True,
    ) -> None:
        """
        Initialize Sparse Variational Wishart Process (SVWP) model.

        The model needs to be instructed about the number of latent GPs by passing `num_latent_gps=likelihood.latent_dim`.

        Parameters
        ----------
        :param D:
            Number of time series, e.g. the number of brain Volumes of Interest (VOI).
        :param Z:
            Initial variational inducing points, of shape (n_inducing_points, 1).
        :param nu:
            Degrees of freedom.
        :param kernel:
        :param num_mc_samples:
            Number of Monte Carlo samples taken to approximate the ELBO.
        :param A_scale_matrix_option:
        :param train_additive_noise:
        :param kernel_lengthscale_init:
        :param q_sqrt_init:
        :param num_factors:
            Number of factors to use in the factored model.
            If None, the non-factored model will be instantiated.
        :param verbose:
        """
        self.D = D
        logging.info(f"Found {self.D:d} time series (D = {self.D:d}).")

        assert len(Z.shape) == 2

        if nu is None:
            nu = self.D
        self.nu = nu

        if num_factors is not None:
            likel = FactoredWishartProcessLikelihood()
        else:
            likel = WishartProcessLikelihood(
                D=self.D,
                nu=nu,
                num_mc_samples=num_mc_samples,
                A_scale_matrix_option=A_scale_matrix_option,
                train_additive_noise=train_additive_noise,
                verbose=verbose,
            )
            assert likel.latent_dim == self.D * nu
        super().__init__(
            kernel=kernel,
            likelihood=likel,
            inducing_variable=Z,
            num_latent_gps=likel.latent_dim,  # number of outputs (multi−output GP)
        )
        self._initialize_parameters(
            kernel_lengthscale_init=kernel_lengthscale_init,
            q_sqrt_init=q_sqrt_init,
        )

    def predict_cov(
            self,
            x_new: np.array,
            num_mc_samples: int = 300,
    ) -> (tf.Tensor, tf.Tensor):
        """
        The main attribute to predict covariance matrices at any point in time.

        Parameters
        ----------
        :param x_new:
            The test locations where we want to predict the covariance matrices.
            Array of shape (x_new, 1).
        :param num_mc_samples:
        :return:
            Tuple of (mean, stddev) of covariance matrices.
        """
        cov_samples = self._get_cov_samples(x_new, num_mc_samples)  # (S_new, N_new, D, D)

        cov_mean = tf.math.reduce_mean(cov_samples, axis=0)  # (N_new, D, D)
        assert are_all_positive_definite(cov_mean)
        cov_stddev = tf.math.reduce_std(cov_samples, axis=0)  # (N_new, D, D)

        return cov_mean, cov_stddev

    def predict_cov_samples(
            self, x_new: np.array, num_mc_samples: int = 300
    ) -> tf.Tensor:
        """
        TODO: we don't use this

        The main attribute to predict covariance matrices at any point in time.

        Parameters
        ----------
        :param x_new:
        :param num_mc_samples:
        :return:
        """
        cov_samples = self._get_cov_samples(x_new, num_mc_samples)  # (S_new, N_new, D, D)
        return cov_samples

    def predict_corr(
            self,
            x_new: np.array,
            num_mc_samples: int = 300,
    ) -> (tf.Tensor, tf.Tensor):
        """
        The main attribute to predict correlation matrices at any point in time.

        TODO: how should we convert sampling uncertainty to correlation confidence interval?

        Parameters
        ----------
        :param x_new:
        :param num_mc_samples:
        :return:
            Tuple of (mean, stddev) of correlation matrices.
        """
        cov_samples = self._get_cov_samples(x_new, num_mc_samples)  # (S_new, N_new, D, D)

        corr_samples = convert_tensor_to_correlation(cov_samples)  # (S_new, N_new, D, D)

        corr_mean = tf.math.reduce_mean(corr_samples, axis=0)  # (N_new, D, D)
        corr_stddev = tf.math.reduce_std(corr_samples, axis=0)  # (N_new, D, D)

        return corr_mean, corr_stddev

    def _get_cov_samples(
            self, x_new: np.array, num_mc_samples: int = 300
    ) -> tf.Tensor:
        """
        Prediction routine for covariance matrices.
        We could switch to a MLE routine for this too, i.e. removing the dependency on Monte Carlo samples.
        But in the Bayesian setting the sampling should work well.

        Parameters
        ----------
        :param x_new:
            The (new) locations to get covariance matrix for.
            Can be different from those in training data.
        :param num_mc_samples:
            S_new: number of Monte Carlo samples to approximate covariance matrix with.
            300 samples are used in previous paper.
        :return:
            AFFA here are the constructed covariance matrix samples.
        """
        num_test_time_steps = x_new.shape[0]

        # TODO: maybe we can get f_samples instead of f?
        f_mean_new, f_variance_new = self.predict_f(x_new)  # (N_new, D * nu), (N_new, D * nu)
        f_mean_new = tf.reshape(f_mean_new, [num_test_time_steps, self.D, -1])  # (N_new, D, nu)
        f_variance_new = tf.reshape(f_variance_new, [num_test_time_steps, self.D, -1])  # (N_new, D, nu)
        f_stddev_new = f_variance_new ** 0.5  # (N_new, D, nu)

        f_sample = tf.random.normal(
            (num_mc_samples, num_test_time_steps, self.D, self.nu),
            mean=0.0, stddev=1.0,
            dtype=tf.dtypes.float64
        ) * f_stddev_new + f_mean_new  # (S_new, N_new, D, nu)

        # TODO: does this still work if nu != D?
        # print(self.likelihood.A_scale_matrix)
        # af = tf.matmul(self.likelihood.A_scale_matrix, f_sample)  # (S_new, N_new, D, nu)
        af = tf.multiply(self.likelihood.A_scale_matrix, f_sample)  # (S_new, N_new, D, nu)
        affa = tf.matmul(af, af, transpose_b=True)  # (S_new, N_new, D, D)
        affa = self.likelihood._add_diagonal_additive_noise(affa)  # (S_new, N_new, D, D)

        return affa

    def scale_matrix(self):
        return self.likelihood.A_scale_matrix * self.likelihood.A_scale_matrix.T

    def _initialize_parameters(
            self, kernel_lengthscale_init: float, q_sqrt_init: float
    ) -> None:
        """
        Set initial values of trainable parameters.

        Parameters
        ----------
        :param kernel_lengthscale_init:
        :param q_sqrt_init:
        """
        self.kernel.lengthscales.assign(kernel_lengthscale_init)
        # m.q_mu.assign(m.q_mu + 0.5)
        self.q_sqrt.assign(self.q_sqrt * q_sqrt_init)

    def save_model_params_dict(
            self, savedir: str, model_name: str
    ) -> None:
        """
        We only save the trained model parameters.
        At loading time, we re-instantiate the model and assign the saved model parameters.
        All tensors need to be converted to floats or lists before saving, as ndarrays are not JSON serializable.
        This currently only works for our Matern52 kernel!

        Parameters
        ----------
        :param savedir:
        :param model_name:
            A string that ends in `.json`.
        """
        params_dict = {
            'D': self.D,
            'nu': self.nu,
            'A_scale_matrix': self.likelihood.A_scale_matrix.numpy().tolist(),
            'additive_noise': self.likelihood.additive_part.numpy().tolist(),
            'kernel_variance': float(self.kernel.variance.numpy()),
            'kernel_lengthscales': float(self.kernel.lengthscales.numpy()),
            'Z': self.inducing_variable.Z.numpy().tolist(),
            'q_mu': self.q_mu.numpy().tolist(),
            'q_sqrt': self.q_sqrt.numpy().tolist(),  # (D*nu, N, N) or (D*nu, Z, Z)
        }
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir, model_name), 'w') as fp:
            json.dump(params_dict, fp)
        logging.info(f"Model '{model_name:s}' saved in '{savedir:s}'.")

    def load_from_params_dict(
            self, savedir: str, model_name: str
    ) -> None:
        """
        This assumes you have created a new model.
        All trained model parameters are assigned.

        Parameters
        ----------
        :param savedir:
        :param model_name:
        """
        with open(os.path.join(savedir, model_name), 'r') as fp:
            params_dict = json.load(fp)

        A_scale_matrix = np.array(params_dict['A_scale_matrix'])
        # print('Loaded A scale matrix:', A_scale_matrix)
        if A_scale_matrix.ndim == 1:
            A_scale_matrix = A_scale_matrix * np.eye(len(A_scale_matrix))
            print('Squared A scale matrix:', A_scale_matrix)
        self.likelihood.A_scale_matrix.assign(A_scale_matrix)

        additive_part = np.array(params_dict['additive_noise'])
        self.likelihood.additive_part.assign(additive_part)

        kernel_variance = params_dict['kernel_variance']
        self.kernel.variance.assign(kernel_variance)

        kernel_lengthscales = params_dict['kernel_lengthscales']
        self.kernel.lengthscales.assign(kernel_lengthscales)

        Z = np.array(params_dict['Z'])
        self.inducing_variable.Z.assign(Z)

        q_mu = np.array(params_dict['q_mu'])
        self.q_mu.assign(q_mu)

        q_sqrt = np.array(params_dict['q_sqrt'])
        self.q_sqrt.assign(q_sqrt)

        logging.info(f"SVWP model '{model_name:s}' loaded from '{savedir:s}'.")
