import logging
import numpy as np
from numpy.testing import assert_array_equal
import unittest

import fcest

from fcest.models.sliding_windows import SlidingWindows

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestSlidingWindows(unittest.TestCase):

    def test_static_covariance_estimate_bivariate(self):
        y_test = np.array([
            [0, 2],
            [1, 1],
            [2, 0]
        ])  # (N, D)
        n_time_steps = y_test.shape[0]
        x_test = np.linspace(0, 1, n_time_steps)
        sw = SlidingWindows(
            x_train_locations=x_test,
            y_train_locations=y_test
        )
        cov_estimates = sw.estimate_static_functional_connectivity(connectivity_metric='covariance')
        true_cov = np.array([
            [1., -1.],
            [-1., 1.]
        ])
        true_cov = np.tile(true_cov, reps=(n_time_steps, 1, 1))
        assert_array_equal(cov_estimates, true_cov)

    def test_static_correlation_estimate_bivariate(self):
        y_test = np.array([
            [0, 2],
            [1, 1],
            [2, 0]
        ])  # (N, D)
        n_time_steps = y_test.shape[0]
        x_test = np.linspace(0, 1, n_time_steps)
        sw = SlidingWindows(
            x_train_locations=x_test,
            y_train_locations=y_test
        )
        corr_estimates = sw.estimate_static_functional_connectivity(connectivity_metric='correlation')
        true_corr = np.array([
            [1., -1.],
            [-1., 1.]
        ])
        true_corr = np.tile(true_corr, reps=(n_time_steps, 1, 1))
        assert_array_equal(corr_estimates, true_corr)

    def test_static_covariance_estimate_trivariate(self):
        y_test = np.array([
            [0, 2, 2],
            [1, 1, 1],
            [2, 0, 0]
        ])  # (N, D)
        n_time_steps = y_test.shape[0]
        x_test = np.linspace(0, 1, n_time_steps)
        sw = SlidingWindows(
            x_train_locations=x_test,
            y_train_locations=y_test
        )
        cov_estimates = sw.estimate_static_functional_connectivity(connectivity_metric='covariance')
        true_cov = np.array([
            [1., -1., -1.],
            [-1., 1., 1.],
            [-1., 1., 1.]
        ])
        true_cov = np.tile(true_cov, (n_time_steps, 1, 1))
        assert_array_equal(cov_estimates, true_cov)


if __name__ == "__main__":
    unittest.main()
