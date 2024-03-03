import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "highpass_filter_data", 
    "butter_highpass_filter", 
    "_butter_highpass", 
    "_compute_lower_frequency_cutoff",
]


def highpass_filter_data(
        y_observed: np.array,
        window_length: int,
        repetition_time: float,
) -> np.array:
    """
    We want to remove frequencies below 1 / window length (in seconds).

    References:
        https://nilearn.github.io/stable/modules/generated/nilearn.signal.butterworth.html

    Parameters
    ----------
    :param y_observed:
        Array of shape (N, D).
    :param window_length:
        In TRs.
    :param repetition_time:
        TR in seconds.
    :return:
    """
    sampling_rate = 1.0 / repetition_time
    nyquist_frequency = sampling_rate / 2

    y_filtered = np.zeros_like(y_observed)
    for i_time_series, single_time_series in enumerate(y_observed.T):
        y_filtered[:, i_time_series] = butter_highpass_filter(
            single_time_series,
            cutoff_low=_compute_lower_frequency_cutoff(window_length, repetition_time),
            nyquist_freq=nyquist_frequency,
        )
    return y_filtered


def butter_highpass_filter(data, cutoff_low, nyquist_freq) -> np.array:
    """
    Apply a high-pass filter to the data.

    Parameters
    ----------
    :param data:
        Array of shape (N, ).
    :param cutoff_low:
        The cutoff frequency of the filter (in Hz).
    :param nyquist_freq:
        The Nyquist frequency of the data (in Hz).
    :return:
        Filtered time series of shape (N, ).
    """
    b, a = _butter_highpass(
        cutoff_low,
        nyquist_freq=nyquist_freq,
    )
    filtered_data = filtfilt(
        b,
        a,
        data,
    )
    return filtered_data


def _butter_highpass(cutoff_low, nyquist_freq, order=5) -> tuple[np.array, np.array]:
    """
    Design a high-pass filter.

    Parameters
    ----------
    :param cutoff_low:
        The cutoff frequency of the filter (in Hz).
    :param nyquist_freq:
        The Nyquist frequency of the data (in Hz).
    :param order:
    :return:
    """
    normal_cutoff_low = cutoff_low / nyquist_freq
    b, a = butter(
        order,
        normal_cutoff_low,
        btype='high',
        analog=False,
    )
    return b, a


def _compute_lower_frequency_cutoff(
        window_length,
        repetition_time,
) -> float:
    """
    Compute the lower frequency cutoff for the high-pass filter.
    
    Parameters
    ----------
    :param window_length:
        In TRs.
    :param repetition_time:
        In seconds.
    :return:
        Float of lower frequency cutoff.
    """
    window_length_in_seconds = window_length * repetition_time
    return 1 / window_length_in_seconds
