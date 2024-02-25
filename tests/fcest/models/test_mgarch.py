import logging
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from fcest.models.mgarch import MGARCH

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestMGARCH(unittest.TestCase):
    """
    Test the MGARCH class.
    """

    def test_mgarch(self):
        """
        Test various instantiations of the MGARCH class.
        """

        # In the bivariate case (D = 2), joint and pairwise estimation should be equivalent.
        m = MGARCH(
            mgarch_type='DCC'
        )
        m.fit_model(
            training_data_df=self._get_dummy_training_data(),
            training_type='joint'
        )
        cov_structure_joint = m.train_location_covariance_structure
        m.fit_model(
            training_data_df=self._get_dummy_training_data(),
            training_type='pairwise'
        )
        cov_structure_pairwise = m.train_location_covariance_structure
        assert_array_almost_equal(cov_structure_joint, cov_structure_pairwise)

    @staticmethod
    def _get_dummy_training_data(
        num_time_series: int = 2, num_time_steps: int = 400
    ) -> pd.DataFrame:
        np.random.seed(2023)
        return pd.DataFrame(
            np.random.normal(size=(num_time_steps, num_time_series))
        )


if __name__ == "__main__":
    unittest.main()
