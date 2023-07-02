import logging
import unittest

import tensorflow as tf

from fcest.helpers.array_operations import are_all_positive_definite

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


if __name__ == "__main__":
    unittest.main()
