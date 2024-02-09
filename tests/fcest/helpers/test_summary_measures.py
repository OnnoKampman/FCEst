import unittest

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from fcest.helpers.summary_measures import summarize_tvfc_estimates


class TestSummaryMeasures(unittest.TestCase):
    """
    Test functions in summary_measures.py.
    """

    def test_summarize_tvfc_estimates(self):
        test_covariance_structure = self._simulate_d2_time_series()
        test_summarized_covariance_structure = summarize_tvfc_estimates(
            full_covariance_structure=test_covariance_structure,
            tvfc_summary_metric='ar1'
        )

        self.assertEqual(len(test_summarized_covariance_structure.shape), 2)
        self.assertAlmostEqual(
            test_summarized_covariance_structure[0, 1], 0.9,
            places=1
        )

        test_covariance_structure = self._simulate_d3_time_series()
        test_summarized_covariance_structure = summarize_tvfc_estimates(
            full_covariance_structure=test_covariance_structure,
            tvfc_summary_metric='ar1'
        )

        self.assertEqual(len(test_summarized_covariance_structure.shape), 2)
        self.assertAlmostEqual(
            test_summarized_covariance_structure[0, 1], -0.9,
            places=1
        )
        self.assertAlmostEqual(
            test_summarized_covariance_structure[0, 2], -0.9,
            places=1
        )
        self.assertAlmostEqual(
            test_summarized_covariance_structure[1, 2], -0.9,
            places=1
        )

    @staticmethod
    def _simulate_d2_time_series() -> np.array:
        """
        Get dummy time series.
        This has unit variances, so that the correlation structure is identical to it.
        :return: array of shape (N, D, D)
        """
        N = 1200
        D = 2

        # D = 2: AR parameter = +0.9
        ar = np.array([1, -0.9])
        ma = np.array([1])
        ar_object = ArmaProcess(ar, ma)

        ts = np.ones(shape=(N, D, D))
        ts[:, 0, 1] = ts[:, 1, 0] = ar_object.generate_sample(nsample=N)

        return ts

    @staticmethod
    def _simulate_d3_time_series() -> np.array:
        """
        Get dummy time series.
        This has unit variances, so that the correlation structure is identical to it.
        :return: array of shape (N, D, D)
        """
        N = 1200
        D = 3

        # D = 3 time series: AR parameter = -0.9
        ar = np.array([1, 0.9])
        ma = np.array([1])
        ar_object = ArmaProcess(ar, ma)

        ts = np.ones(shape=(N, D, D))
        ts[:, 0, 1] = ts[:, 1, 0] = ar_object.generate_sample(nsample=N)
        ts[:, 0, 2] = ts[:, 2, 0] = ar_object.generate_sample(nsample=N)
        ts[:, 1, 2] = ts[:, 2, 1] = ar_object.generate_sample(nsample=N)

        return ts


if __name__ == '__main__':
    unittest.main()
