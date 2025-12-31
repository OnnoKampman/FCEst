import logging
import unittest

import numpy as np

from fcest.features.brain_states import BrainStatesExtractor

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestBrainStatesExtractor(unittest.TestCase):
    """
    Test functions in ./features/brain_states.py.
    """

    def test_brain_states_extractor_initialization(self):
        """
        Test BrainStatesExtractor initialization.
        """
        brain_states_extractor = BrainStatesExtractor(
            connectivity_metric='correlation',
            num_time_series=2,
            tvfc_estimates=np.random.random(size=(7, 5, 3)),
        )


if __name__ == "__main__":
    unittest.main()
