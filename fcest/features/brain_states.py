import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans

from fcest.helpers.array_operations import reconstruct_symmetric_matrix_from_tril

if TYPE_CHECKING:
    pass

__all__ = ["BrainStatesExtractor"]


class BrainStatesExtractor:

    def __init__(
        self,
        connectivity_metric: str,
        num_time_series: int,
        tvfc_estimates: npt.NDArray[np.float64],
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
    ) -> tuple[float, pd.DataFrame, pd.DataFrame]:
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

    def compute_basis_state(
        self,
        all_subjects_tril_tvfc: npt.NDArray[np.float64],
        num_basis_states: int,
        num_time_series: int,
        num_time_steps: int,
        brain_states_savedir: str,
    ) -> tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        Brain states (recurring whole-brain patterns) are a way to summarize estimated TVFC.
        Here we follow Allen et al. (2012) and use k-means clustering to find the brain states across subjects,
        independently for each of the two runs within each of the two session.

        Parameters
        ----------
        :param all_subjects_tril_tvfc:
            Array of shape (num_subjects * N, D*(D-1)/2).
        :param num_basis_states: 
            The number of states/clusters we want to extract.
            Ideally this number would be determined automatically, or based on some k-means elbow analysis.
        :param num_time_series:
        :param num_time_steps:
        :param brain_states_savedir:
            Directory to save brain states to.
        :return:
        """
        logging.info("Running k-means clustering...")
        kmeans = KMeans(
            n_clusters=num_basis_states,
            algorithm='lloyd',
            n_init=10,
            verbose=0,
        ).fit(all_subjects_tril_tvfc)
        logging.info("Finished k-means clustering.")

        cluster_centers = self._get_cluster_centroids(
            kmeans=kmeans,
            num_time_series=num_time_series
        )

        cluster_centers, cluster_sort_order = self._sort_cluster_centers(
            cluster_centers=cluster_centers
        )

        # Save clusters (i.e. brain states) to file.
        brain_states_path = Path(brain_states_savedir)
        brain_states_path.mkdir(parents=True, exist_ok=True)
        for i_cluster, cluster_centroid in enumerate(cluster_centers):
            cluster_df = pd.DataFrame(cluster_centroid)  # (D, D)
            cluster_df.to_csv(
                brain_states_path / f'{self.connectivity_metric:s}_brain_state_{i_cluster:d}.csv',
                float_format='%.2f',
            )
            logging.info(f"Brain state saved in '{brain_states_savedir:s}'.")

        all_subjects_brain_state_assignments_df = self._get_brain_state_assignments(
            labels=kmeans.labels_,
            num_time_steps=num_time_steps
        )  # (num_subjects, N)

        all_subjects_brain_state_assignments_df = all_subjects_brain_state_assignments_df.replace(
            to_replace=cluster_sort_order,
            value=np.arange(num_basis_states)
        )

        all_subjects_dwell_times_df = self._compute_dwell_time(
            num_brain_states=num_basis_states,
            brain_state_assignments=all_subjects_brain_state_assignments_df
        )  # (num_subjects, num_brain_states)

        return kmeans.inertia_, all_subjects_brain_state_assignments_df, all_subjects_dwell_times_df

    def _get_cluster_centroids(
        self,
        kmeans: KMeans,
        num_time_series: int,
    ) -> npt.NDArray[np.float64]:
        """
        Get cluster centroids (centers) from k-means clustering.
        """
        # Get cluster centers - these are the characteristic basis brain states.
        cluster_centers = kmeans.cluster_centers_  # (num_clusters, num_features)

        # Reconstruct correlation matrix per cluster.
        cluster_centers = [
            reconstruct_symmetric_matrix_from_tril(cluster_vector, num_time_series) for cluster_vector in cluster_centers
        ]
        cluster_centers = np.array(cluster_centers)  # (num_clusters, D, D)

        return cluster_centers

    def _sort_cluster_centers(
        self,
        cluster_centers: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:

        # Re-order clusters based on high to low (descending) 'contrast' - higher contrast states are more interesting.
        cluster_contrasts = np.var(cluster_centers, axis=(1, 2))  # (num_clusters, )
        cluster_sort_order = np.argsort(cluster_contrasts)[::-1]  # (num_clusters, )

        cluster_centers = cluster_centers[cluster_sort_order, :, :]  # (num_clusters, D, D)

        return cluster_centers, cluster_sort_order

    def _get_brain_state_assignments(
        self,
        labels: npt.NDArray[np.int32],
        num_time_steps: int,
    ) -> pd.DataFrame:
        """
        Get labels per covariance matrix, which can be used to re-construct a states time series per subject.
        Each subject is now only characterized by an assignment to one of the clusters at each time step.

        Parameters
        ----------
        :param labels: 
            Array of shape (num_subjects * N, ).
        :param num_time_steps:
        :return:
        """
        assert len(labels) % num_time_steps == 0
        num_subjects = int(len(labels) / num_time_steps)

        labels = labels.reshape(num_subjects, num_time_steps)  # (num_subjects, N)

        labels_df = pd.DataFrame(labels)

        return labels_df
