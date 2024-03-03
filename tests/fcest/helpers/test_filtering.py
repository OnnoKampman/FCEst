import unittest

import numpy as np

from fcest.helpers.filtering import highpass_filter_data, butter_highpass_filter


class TestFiltering(unittest.TestCase):
    """
    Test functions in filtering.py.
    """

    def test_highpass_filter_data(self):
        """
        Test that the function returns a NumPy array.
        """
        filtered_data = highpass_filter_data(
            y_observed=self._simulate_d2_time_series()[1],
            window_length=10,
            repetition_time=2,
        )
        self.assertEqual(type(filtered_data), np.ndarray)

    def test_butter_highpass_filter(self):
        """
        Test that the function returns a NumPy array.
        """
        filtered_data = butter_highpass_filter(
            data=self._simulate_d2_time_series()[1][:, 0],
            cutoff_low=0.1,
            nyquist_freq=0.5,
        )
        self.assertEqual(type(filtered_data), np.ndarray)

    @staticmethod
    def _simulate_d2_time_series() -> np.array:
        """
        Get dummy time series.

        :return:
            Array of shape (N, D)
        """
        N = 200
        D = 2
        x = np.linspace(0, 1, N).reshape(-1, 1)
        y = np.random.random(size=(N, D))

        return x, y


if __name__ == '__main__':
    unittest.main()
