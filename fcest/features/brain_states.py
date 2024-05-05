import logging
import os
import sys

from fcest.helpers.data import to_3d_format
import math
from nilearn import connectome
import numpy as np
import pandas as pd
from scipy import cluster
from sklearn.cluster import KMeans

from helpers.array_operations import reconstruct_symmetric_matrix_from_tril
from helpers.brain_states import compute_basis_state, extract_number_of_brain_state_switches
from helpers.icc import compute_icc_scores_pingouin


class BrainStatesExtractor:

    def __init__(
        self,
        connectivity_metric: str,
        num_time_series: int,
        tvfc_estimates: np.array,
    ) -> None:
        """
        Class for extracting brain states from TVFC estimates.

        Parameters
        ----------
        connectivity_metric : str, default='correlation'
        tvfc_estimates : np.array
            Array of shape (num_subjects, num_time_steps, num_features).
        """
        logging.info("Initializing BrainStatesExtractor...")

        self.connectivity_metric = connectivity_metric
        self.num_time_steps = tvfc_estimates.shape[1]
        self.num_time_series = num_time_series

        self.tvfc_estimates_tril = tvfc_estimates.reshape(-1, tvfc_estimates.shape[-1])

    def extract_brain_states(
        self,
        num_brain_states: int,
    ):
        """
        Extract brain states from TVFC estimates.

        Parameters
        ----------
        num_brain_states : int
        """
        return self.compute_basis_state(
            all_subjects_tril_tvfc=self.tvfc_estimates_tril,
            num_basis_states=num_brain_states,
            num_time_series=self.num_time_series,
            num_time_steps=self.num_time_steps,
        )
