import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from fcest.helpers.array_operations import to_correlation_structure


class TestArrayOperations(unittest.TestCase):
    """Test functions in array_operations.py."""

    def test_to_correlation_structure(self):
        # test identical covariance and correlation structures
        test_covariance_structure = self._get_test_covariance_structure()
        test_correlation_structure = to_correlation_structure(test_covariance_structure)
        assert_array_almost_equal(test_correlation_structure, test_covariance_structure)

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
