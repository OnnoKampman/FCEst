import logging

import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from .array_operations import get_all_lower_triangular_indices_tuples

__all__ = [
    "summarize_tvfc_estimates", 
    "fit_and_extract_ar1_param", 
    "compute_rate_of_change", 
    "_rate_of_change",
]


def summarize_tvfc_estimates(
        full_covariance_structure: np.array, tvfc_summary_metric: str
) -> np.array:
    """
    Summarize a full TVFC covariance structure over temporal axis.

    Parameters
    ----------
    :param full_covariance_structure:
        TVFC array of shape (N, D, D).
    :param tvfc_summary_metric:
    :return:
        array of shape (D, D)
    """
    match tvfc_summary_metric:
        case 'ar1':
            return fit_and_extract_ar1_param(full_covariance_structure)  # (D, D)
        case 'fourier_max':
            raise NotImplementedError
        case 'fourier_mean':
            raise NotImplementedError
        case 'mean':
            return np.nanmean(full_covariance_structure, axis=0)  # (D, D)
        case 'rate_of_change':
            return compute_rate_of_change(full_covariance_structure)  # (D, D)
        case 'std':
            return np.nanstd(full_covariance_structure, axis=0)  # (D, D)
        case 'variance':
            return np.nanvar(full_covariance_structure, axis=0)  # (D, D)
        case _:
            logging.error(f"TVFC summary metric {tvfc_summary_metric:s} not recognized.")


def fit_and_extract_ar1_param(full_covariance_structure: np.array) -> np.array:
    """
    Summarize estimated TVFC by taking AR(1) component of each time series.
    Diagonal terms are set to zero at the moment.
    """
    indices = get_all_lower_triangular_indices_tuples(
        num_time_series=full_covariance_structure.shape[1]
    )

    ar1_coefficients = np.zeros(
        shape=(full_covariance_structure.shape[1], full_covariance_structure.shape[2])
    )
    for tup_i, tup_j in indices:
        model = AutoReg(
            full_covariance_structure[:, tup_i, tup_j],
            lags=1
        )
        model_fit = model.fit()

        # print(model_fit.summary())
        # print('Coefficients: %s' % model_fit.params)

        ar1_coefficients[tup_i, tup_j] = ar1_coefficients[tup_j, tup_i] = model_fit.params[-1]

    return ar1_coefficients


def compute_rate_of_change(full_covariance_structure: np.array) -> np.array:
    """
    Rate of change is a time series summary statistic that measures the intensity of fluctuations in time.
    It is defined here as the average relative step size across time.

    Parameters
    ----------
    :param full_covariance_structure: array of shape (N, D, D).
    :return:
        array of shape (D, D)
    """
    num_time_series = full_covariance_structure.shape[1]  # D

    average_rate_of_change = np.zeros(shape=(num_time_series, num_time_series))  # (D, D)
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

    Parameters
    ----------
    :param current_value: 2D array
    :param previous_value: 2D array
    :return:
        2D array with rates of change.
    """
    rate_of_change = np.abs(current_value / previous_value - 1)
    rate_of_change[rate_of_change == np.inf] = 1
    rate_of_change[rate_of_change > 10] = 1
    return rate_of_change
