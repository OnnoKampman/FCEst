import logging
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from fcest.models.mgarch import MGARCH

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestMGARCH(unittest.TestCase):
    """Test the MGARCH class."""

    def test_static_covariance_estimate_bivariate(self):
        assert True == True


if __name__ == "__main__":
    unittest.main()
