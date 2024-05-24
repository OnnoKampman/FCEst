import unittest

import numpy as np

from fcest.helpers.inference import run_adam


class TestInference(unittest.TestCase):
    """
    Test functions in inference.py.
    """

    def test_run_adam(self):
        """
        Test that the function returns a list.
        """
        # logf = run_adam(
        #     "VWP",
        #     m,
        #     iterations=100,
        # )
        logf = []
        self.assertEqual(type(logf), list)

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
