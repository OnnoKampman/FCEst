import logging
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf

from fcest.models.likelihoods import WishartProcessLikelihood

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestLikelihoods(unittest.TestCase):
    """
    Test functions in likelihoods.py.
    """

    def test_tensor_computations(
            self, num_mc_samples: int = 2, num_time_steps: int = 6, 
            num_time_series: int = 2, nu: int = 3
    ) -> None:
        """
        Test tensor computations.

        Parameters
        ----------
        :param num_mc_samples:
            Number of MC samples (S).
        :param num_time_steps:
            Number of time steps (N).
        :param num_time_series:
            Number of time series (D).
        :param nu:
            Degrees of freedom.
        """
        A = 2 * np.eye(2)
        A[0, 1] = A[1, 0] = -0.1
        A = tf.Variable(A)
        y_data = np.ones((num_time_steps, num_time_series))

        f_sample = np.random.random(size=(num_time_steps, num_time_series, nu))
        f_sample = np.tile(f_sample[None, :, :, :], [num_mc_samples, 1, 1, 1])
        f_sample = tf.Variable(f_sample)

        af = tf.matmul(A, f_sample)
        affa = tf.matmul(af, af, transpose_b=True)
        L = tf.linalg.cholesky(affa)
        assert_array_almost_equal(affa, tf.matmul(L, L, transpose_b=True))

        log_det_cov = 2 * tf.math.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L)),
            axis=2
        )
        assert_array_almost_equal(log_det_cov, tf.linalg.logdet(affa))

        # Test different ways to compute the same tensor.
        affa_inverse = tf.linalg.inv(affa)  # (S, N, D, D)
        yaffa_1 = tf.einsum('nd,snid->sni', y_data, affa_inverse)  # (S, N, D)
        yaffa_2 = tf.matmul(y_data, affa_inverse)  # (S, N, N, D)
        yaffa_2 = tf.math.reduce_mean(yaffa_2, axis=2)  # (S, N, D)
        assert_array_almost_equal(yaffa_1, yaffa_2)

        # Test different ways to compute the same tensor.
        yaffay_1 = tf.matmul(yaffa_1, y_data, transpose_b=True)
        yaffay_1 = tf.math.reduce_mean(yaffay_1, axis=2)
        y_data_tiled = tf.tile(y_data[None, :, :, None], [num_mc_samples, 1, 1, 1])  # (S, N, D, 1)
        L_solve_y = tf.linalg.triangular_solve(L, y_data_tiled, lower=True)
        yaffay_2 = tf.math.reduce_sum(L_solve_y ** 2, axis=(2, 3))
        assert_array_almost_equal(yaffay_1, yaffay_2)

    def test_wishart_process_likelihood(self):

        wishart_process_likelihood = WishartProcessLikelihood(
            D=2,
            nu=2,
            num_mc_samples=7,
            A_scale_matrix_option='train_full_matrix',
            train_additive_noise=True
        )


if __name__ == "__main__":
    unittest.main()
