import logging
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestLikelihoods(unittest.TestCase):

    def test_tensor_computations(self, S=2, N=6, D=2, nu=3):
        """
        :param S: number of MC samples.
        :param N: number of time steps.
        :param D: number of time series.
        :param nu: degrees of freedom.
        :return:
        """
        A = 2 * np.eye(2)
        A[0, 1] = A[1, 0] = -0.1
        A = tf.Variable(A)
        y_data = np.ones((N, D))

        f_sample = np.random.random(size=(N, D, nu))
        f_sample = np.tile(f_sample[None, :, :, :], [S, 1, 1, 1])
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
        yaffa_1 = tf.einsum('nd,snid->sni', y_data, affa_inverse)
        yaffa_2 = tf.matmul(y_data, affa_inverse)  # (S, N, N, D)
        yaffa_2 = tf.math.reduce_mean(yaffa_2, axis=2)
        assert_array_almost_equal(yaffa_1, yaffa_2)

        # Test different ways to compute the same tensor.
        y_data_tiled = tf.tile(y_data[None, :, :, None], [S, 1, 1, 1])
        yaffay_1 = tf.matmul(yaffa_1, y_data, transpose_b=True)
        yaffay_1 = tf.math.reduce_mean(yaffay_1, axis=2)
        L_solve_y = tf.linalg.triangular_solve(L, y_data_tiled, lower=True)
        yaffay_2 = tf.math.reduce_sum(L_solve_y ** 2, axis=(2, 3))
        assert_array_almost_equal(yaffay_1, yaffay_2)


if __name__ == "__main__":
    unittest.main()
