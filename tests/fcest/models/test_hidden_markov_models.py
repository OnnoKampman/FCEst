import logging
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from fcest.models.hidden_markov_models import HiddenMarkovModel

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestHiddenMarkovModel(unittest.TestCase):
    """
    Test the HiddenMarkovModel class.
    """

    def test_hidden_markov_model(self):
        """
        Test various instantiations of the HiddenMarkovModel class.
        """

        m = HiddenMarkovModel()
        m.fit_model(
            training_data_df=self._get_dummy_training_data(),
        )

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
