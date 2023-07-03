import unittest

import numpy as np

from fcest.helpers.data import to_2d_format, to_3d_format


class TestData(unittest.TestCase):
    """Test functions in data.py."""

    def test_to_2d_format(self):
        test_3d_format_covariance_structure = self._get_test_covariance_structure()
        test_2d_format_covariance_structure = to_2d_format(test_3d_format_covariance_structure)
        self.assertEqual(len(test_2d_format_covariance_structure.shape), 2)

    def test_to_3d_format(self):
        test_3d_format_covariance_structure = self._get_test_covariance_structure()
        test_2d_format_covariance_structure = to_2d_format(test_3d_format_covariance_structure)
        test_3d_format_covariance_structure = to_3d_format(test_2d_format_covariance_structure)
        self.assertEqual(len(test_3d_format_covariance_structure.shape), 3)

    @staticmethod
    def _get_test_covariance_structure() -> np.array:
        """
        Get dummy covariance structure.
        This has unit variances, so that the correlation structure is identical to it.
        """
        return np.array([
            [[1.0, 0.3, 0.5],
             [0.3, 1.0, 0.2],
             [0.5, 0.2, 1.0]],
            [[1.0, 0.1, 0.4],
             [0.1, 1.0, 0.3],
             [0.4, 0.3, 1.0]]
        ])


if __name__ == '__main__':
    unittest.main()
