import logging

import numpy as np


def summarize_tvfc_estimates(
        full_covariance_structure: np.array, tvfc_summary_metric: str
) -> np.array:
    """
    Summarize a full TVFC covariance structure.
    :param full_covariance_structure: TVFC array of shape (N, D, D)
    :param tvfc_summary_metric:
    :return:
        array of shape (D, D)
    """
    match tvfc_summary_metric:
        case 'mean':
            return np.nanmean(full_covariance_structure, axis=0)  # (D, D)
        case 'variance':
            return np.nanvar(full_covariance_structure, axis=0)  # (D, D)
        case 'std':
            return np.nanstd(full_covariance_structure, axis=0)  # (D, D)
        case 'rate_of_change':
            return compute_rate_of_change(full_covariance_structure)  # (D, D)
        case 'fourier_mean':
            raise NotImplementedError
        case 'fourier_max':
            raise NotImplementedError
        case _:
            logging.error(f"TVFC summary metric {tvfc_summary_metric:s} not recognized.")


def compute_rate_of_change(full_covariance_structure: np.array) -> np.array:
    """
    Rate of change is a time series summary statistic that measures the intensity of fluctuations in time.
    It is defined here as the average relative step size across time.
    :param full_covariance_structure: array of shape (N, D, D).
    :return:
        array of shape (D, D)
    """
    n_time_series = full_covariance_structure.shape[1]  # D

    average_rate_of_change = np.zeros(shape=(n_time_series, n_time_series))  # (D, D)
    n_changes = full_covariance_structure.shape[0] - 1
    for i_time_step in np.arange(n_changes):
        roc = _rate_of_change(
            current_value=full_covariance_structure[i_time_step+1, :, :],
            previous_value=full_covariance_structure[i_time_step, :, :]
        )
        average_rate_of_change += roc
    average_rate_of_change /= n_changes
    return average_rate_of_change


def _rate_of_change(
        current_value: np.array, previous_value: np.array
) -> np.array:
    """
    TODO: what should the rate of change be when the previous value is a zero?
    :param current_value: 2D array
    :param previous_value: 2D array
    :return:
        2D array with rates of change.
    """
    rate_of_change = np.abs(current_value / previous_value - 1)
    rate_of_change[rate_of_change == np.inf] = 1
    rate_of_change[rate_of_change > 10] = 1
    return rate_of_change
