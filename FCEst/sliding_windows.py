import logging
import numpy as np
import os
import pandas as pd
from scipy.stats import multivariate_normal

from helpers.array_operations import to_correlation_structure
from helpers.data import to_3d_format
from helpers.filtering import highpass_filter_data


class SlidingWindows:
    """
    Main class for sliding-windows methods.
    TODO: implement tapered sliding-windows (TSW)
    """

    def __init__(
            self, x_train_locations: np.array, y_train_locations: np.array,
            repetition_time: float = None, window_shape: str = 'rectangle'
    ):
        """
        Main sliding-windows functional connectivity class.
        :param x_train_locations: array of expected shape (n_time_steps, 1), i.e. (N, 1)
        :param y_train_locations: array of expected shape (n_time_steps, n_time_series), i.e. (N, D)
        :param repetition_time: if no repetition time is fed, then we assume we are running on synthetic data.
        :param window_shape: 'rectangle', 'tapered' (Gaussian)
        :return:
        """
        self.x_train = x_train_locations
        self.y_train = y_train_locations
        assert len(x_train_locations) == len(y_train_locations)

        self.n_time_steps = y_train_locations.shape[0]  # N
        self.n_time_series = y_train_locations.shape[1]  # D

        if repetition_time is None:
            logging.info("No repetition time found, assuming we are running simulations.")
            repetition_time = 1.0
        self.repetition_time = repetition_time
        self._set_min_max_proposed_window_length()

        if window_shape == 'tapered':
            logging.error("Tapered sliding windows not implemented yet.")
        elif window_shape != 'rectangle':
            logging.error("Other window types not implemented yet.")

    def estimate_static_functional_connectivity(self, connectivity_metric: str) -> np.array:
        """
        Static functional connectivity (sFC) estimate.
        :param connectivity_metric:
        :return:
            covariance structure array of shape (N_train, D, D).
        """
        match connectivity_metric:
            case 'covariance':
                sfc_estimate = np.cov(self.y_train.T)  # (D, D)
            case 'correlation':
                sfc_estimate = np.corrcoef(self.y_train.T)  # (D, D)
            case _:
                raise NotImplementedError(f"Connectivity metric {connectivity_metric:s} not recognized.")
        sfc_estimate = np.tile(sfc_estimate, reps=(self.n_time_steps, 1, 1))  # (N, D, D)
        assert len(sfc_estimate.shape) == 3
        return sfc_estimate

    def overlapping_windowed_cov_estimation(
            self, window_length: int,
            step_size: int = 1, repetition_time: float = None, connectivity_metric: str = 'covariance'
    ) -> np.array:
        """
        Overlapping sliding-windows estimate.
        A step size of a single data point is common.
        Typical window lengths range from 30 seconds (15 measurements at TR = 2) to 60 seconds.
        TODO: plot frequency domain before and after filter
        :param window_length: number of observations in one window.
            That is, this is measured in TRs (number of volumes).
            Note: there is still some discussion over what a good window length is.
        :param step_size: number of observations the window is moved.
            Also referred to as window offset.
            A step size of one TR is generally accepted to give the best results.
        :param repetition_time: if no repetition time is fed, then we assume we are running on synthetic data.
        :param connectivity_metric: brain region functional connectivity metric, either 'covariance' or 'correlation'.
        :return:
            TVFC covariance structure array of shape (N, D, D), i.e. (n_time_steps, n_time_series, n_time_series).
        """
        n_time_steps = self.y_train.shape[0]
        n_time_series = self.y_train.shape[1]

        # Highpass filtering as recommended by Leonardi2015.
        if repetition_time is not None:
            y_train = highpass_filter_data(
                self.y_train, window_length=window_length, repetition_time=repetition_time
            )
        else:
            y_train = self.y_train

        # Pad zeros to time series to be able to compute covariances at the edges.
        n_zeros_to_pad = int(np.floor(window_length / 2))
        y_train = np.concatenate((
            np.zeros((n_zeros_to_pad, n_time_series)),
            y_train,
            np.zeros((n_zeros_to_pad, n_time_series))
        ))

        estimated_tvfc = np.zeros((n_time_steps, n_time_series, n_time_series))  # empty covariances array to fill
        for i_window in range(int(n_time_steps / step_size)):
            start_index = int(i_window * step_size)
            y_subset_indices = np.arange(start_index, start_index + window_length)
            y_subset = y_train[y_subset_indices, :]
            estimated_tvfc[i_window, :, :] = np.cov(y_subset.T)

        if connectivity_metric == 'correlation':
            estimated_tvfc = to_correlation_structure(estimated_tvfc)  # (N, D, D)

        return estimated_tvfc

    def find_cross_validated_optimal_window_length(
            self, window_length_step_size: int = 2, eval_location_step_size: int = 1
    ) -> pd.DataFrame:
        """
        This is currently run for a single subject.
        TODO: is this the best way of doing cross-validation?
        TODO: we could also consider this per time point and take the optimal window length estimates as the final estimates.
        TODO: implement a bivariate loop version (each edge has a different window length)
        :param window_length_step_size: should be 2 to make sure the window length is always uneven (and thus symmetrical)
        :param eval_location_step_size:
        :return:
            DataFrame of shape (n_eval_locations, n_proposed_window_lengths).
        """
        n_evaluation_points = self.n_time_steps - self.maximum_proposal_window_length

        proposal_window_length_range = np.arange(
            self.minimum_proposal_window_length, self.maximum_proposal_window_length,
            window_length_step_size
        )
        n_proposal_window_lengths = len(proposal_window_length_range)
        proposal_window_length_range_seconds = proposal_window_length_range * self.repetition_time
        logging.info(f"Proposing {n_proposal_window_lengths:d} window lengths (from {self.minimum_proposal_window_length:d} to {self.maximum_proposal_window_length:d}, in steps of {window_length_step_size:d}).")

        x_eval_location_minimum = int((self.maximum_proposal_window_length - 1) / 2)
        x_eval_location_maximum = self.n_time_steps - int((self.maximum_proposal_window_length + 1) / 2)
        eval_locations = np.arange(
            x_eval_location_minimum, x_eval_location_maximum,
            eval_location_step_size
        )
        n_eval_locations = len(eval_locations)
        logging.info(f"Evaluating on {n_eval_locations:d} observations (from position {x_eval_location_minimum:d} to {x_eval_location_maximum:d}).")

        results_df = pd.DataFrame()
        for proposal_window_length in proposal_window_length_range:
            for x_eval_location in eval_locations:

                # Select single observation we will compute the test likelihood for.
                y_eval_location = self.y_train[x_eval_location, :]  # (D, )

                # Select frame for sliding window.
                test_frame_start = int(x_eval_location - np.floor(proposal_window_length / 2))
                test_frame_end = int(x_eval_location + np.ceil(proposal_window_length / 2))

                # Select testing frame from all observations.
                y_eval_window = self.y_train[:test_frame_end, :]
                y_eval_window = y_eval_window[test_frame_start:, :]  # (proposal_window_length, D)
                assert y_eval_window.shape == (proposal_window_length, self.n_time_series)

                # Remove point of interest.
                new_y_eval_index = int((proposal_window_length - 1) / 2)
                y_eval_window = np.delete(y_eval_window, new_y_eval_index, axis=0)  # (proposal_window_length - 1, D)
                assert y_eval_window.shape == (proposal_window_length - 1, self.n_time_series)

                # Estimated covariance matrix at this point of interest.
                cov_matrix_eval_window = np.cov(y_eval_window.T)  # (D, D)
                assert cov_matrix_eval_window.shape == (self.n_time_series, self.n_time_series)

                # Compute log likelihood of this observation under this covariance matrix.
                mvn_logpdf = multivariate_normal.logpdf(
                    x=y_eval_location,
                    mean=np.zeros(self.n_time_series),
                    cov=cov_matrix_eval_window,
                    allow_singular=True  # TODO: this might be problematic
                )

                # Aggregate results.
                results_df.loc[x_eval_location, proposal_window_length] = mvn_logpdf

        return results_df

    def _set_min_max_proposed_window_length(
            self, min_proposal_window_length_seconds: float = 20.0, 
            max_proposal_window_length_seconds: float = 180.0
    ) -> None:
        """
        Leonardi2015 states that window lengths of 20 seconds would need filtering out of frequencies below 0.05 Hz,
            which are typically of interest in rs-fMRI studies.
        We consider this to be our minimum window length therefore.
        We do not consider a maximum length - if the signal is static then SW approach should be able to learn that.
        TODO: does this work with LEOO data split?
        """
        if self.repetition_time is not None:
            # Find minimum and maximum proposal window lengths in number of TRs.
            minimum_proposal_window_length_trs = int(np.ceil(min_proposal_window_length_seconds / self.repetition_time))
            maximum_proposal_window_length_trs = np.minimum(
                int(np.ceil(self.n_time_steps * self.repetition_time / 2 / self.repetition_time)),
                int(np.ceil(max_proposal_window_length_seconds / self.repetition_time))
            )

            # Convert to odd numbers.
            if minimum_proposal_window_length_trs % 2 == 0:
                minimum_proposal_window_length_trs += 1
            if maximum_proposal_window_length_trs % 2 == 0:
                maximum_proposal_window_length_trs += 1

            self.minimum_proposal_window_length = minimum_proposal_window_length_trs
            self.maximum_proposal_window_length = maximum_proposal_window_length_trs
        else:
            match self.n_time_steps:
                case 60:
                    self.minimum_proposal_window_length = 9
                    self.maximum_proposal_window_length = 31
                case 100:
                    self.minimum_proposal_window_length = 9
                    self.maximum_proposal_window_length = 51
                case 120:
                    self.minimum_proposal_window_length = 9
                    self.maximum_proposal_window_length = 61
                case 200:
                    self.minimum_proposal_window_length = 9
                    self.maximum_proposal_window_length = 121
                case 400:
                    self.minimum_proposal_window_length = 21
                    self.maximum_proposal_window_length = 181
                case 600:
                    self.minimum_proposal_window_length = 21
                    self.maximum_proposal_window_length = 201
                case 1200:
                    self.minimum_proposal_window_length = 21
                    self.maximum_proposal_window_length = 221
                case _:
                    raise NotImplementedError("Number of time steps not recognized for simulations.")

    @staticmethod
    def get_optimal_window_length(results_df: pd.DataFrame) -> int:
        """
        Given a DataFrame of cross-validation results, this returns the optimal window length.
        TODO: we could add a plot of these results for verification
        :param results_df:
        :return:
        """
        # Average over all investigation locations.
        mean_results_df = results_df.mean(axis=0)

        # Get window length where average test likelihood is highest.
        optimal_window_length = mean_results_df.index[mean_results_df.argmax()]

        logging.info(f"Obtained optimal window length: {optimal_window_length:d} TRs.")

        return optimal_window_length

    def windowed_cov_estimation(self, n_windows: int) -> np.array:
        """
        Segmented, non-overlapping windows.
        :param n_windows: number of non-overlapping windows.
            If set to 1, this is the static covariance estimate.
        :return:
            covariance structure array of shape (N, D, D), i.e. (n_time_steps, n_time_series, n_time_series).
        """
        n_time_steps = self.y_train.shape[0]
        n_time_series = self.y_train.shape[1]
        cov_structure = np.zeros((n_time_steps, n_time_series, n_time_series))
        window_length = n_time_steps / n_windows
        for i_window in range(n_windows):
            y_subset_indices = np.arange(int(i_window * window_length), int((i_window + 1) * window_length))
            y_subset = self.y_train[y_subset_indices, :]
            cov_structure[y_subset_indices, :, :] = np.cov(y_subset.T)  # np.cov() accepts multivariate inputs
        return cov_structure

    def save_model_predictions(
            self, optimal_window_length: int, savedir: str, model_name: str,
            repetition_time: float = None, connectivity_metric: str = 'correlation'
    ) -> None:
        """
        Saves TVFC estimates.
        :param optimal_window_length:
        :param savedir:
        :param model_name:
        :param repetition_time:
        :param connectivity_metric: 'correlation', 'covariance'
        """
        train_loc_cov_structure = self.overlapping_windowed_cov_estimation(
            window_length=optimal_window_length,
            repetition_time=repetition_time,
            connectivity_metric=connectivity_metric
        )  # (N_train, D, D)
        cov_structure_df = pd.DataFrame(
            train_loc_cov_structure.reshape(len(train_loc_cov_structure), -1).T
        )  # (D*D, N)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        cov_structure_df.to_csv(
            os.path.join(savedir, model_name)
        )
        logging.info(f"Saved SW-CV model (train location) estimates to '{savedir:s}'.")

    @staticmethod
    def load_model_predictions(
            savedir: str, model_name: str
    ) -> np.array:
        """
        Loads SW-CV model estimates.
        :param savedir:
        :param model_name:
        :return:
            covariance structure array of shape (N, D, D).
        """
        train_loc_cov_structure = pd.read_csv(
            os.path.join(savedir, model_name)
        )  # (D*D, N)
        train_loc_cov_structure = to_3d_format(train_loc_cov_structure)  # (N, D, D)
        return train_loc_cov_structure
