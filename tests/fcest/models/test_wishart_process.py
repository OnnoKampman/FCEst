import logging
import unittest

import numpy as np
import tensorflow as tf

from fcest.helpers.array_operations import are_all_positive_definite
from fcest.models.wishart_process import SparseVariationalWishartProcess, VariationalWishartProcess

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestWishartProcess(unittest.TestCase):

    def test_assert_positive_definite_not_symmetric(self):
        matrices = [
            [
                [1.0, 0.1],
                [0.2, 1.0]
            ],
            [
                [1.0, 0.3],
                [0.4, 1.0]
            ]
        ]
        matrices = tf.constant(matrices, dtype=tf.dtypes.float64)
        self.assertFalse(are_all_positive_definite(matrices))

    def test_assert_positive_definite_symmetric_not_positive_definite(self):
        matrices = [
            [
                [1.0, 2.1],
                [2.1, 1.0]
            ],
            [
                [1.0, 4.3],
                [4.3, 1.0]
            ]
        ]
        matrices = tf.constant(matrices, dtype=tf.dtypes.float64)
        self.assertFalse(are_all_positive_definite(matrices))

    def test_assert_positive_definite_symmetric_positive_definite(self):
        matrices = [
            [
                [1.0, 0.1],
                [0.1, 1.0]
            ],
            [
                [1.0, 0.3],
                [0.3, 1.0]
            ]
        ]
        matrices = tf.constant(matrices, dtype=tf.dtypes.float64)
        self.assertTrue(are_all_positive_definite(matrices))

    def test_sparse_variational_wishart_process(self):
        """Test instantiation of SparseVariationalWishartProcess."""
        n_time_series = 2
        n_time_steps = 7

        m = SparseVariationalWishartProcess(
            D=n_time_series,
            Z=np.arange(n_time_steps),
            nu=n_time_series
        )

    def test_variational_wishart_process(self):
        """Test instantiation of VariationalWishartProcess."""
        n_time_series = 2
        n_time_steps = 7

        m = VariationalWishartProcess(
            x_observed=np.linspace(0, 1, n_time_steps),
            y_observed=np.random.random(size=(n_time_steps, n_time_series)),
            nu=n_time_series
        )


if __name__ == "__main__":
    unittest.main()
