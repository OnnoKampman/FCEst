import numpy as np
import scipy.stats

__all__ = [
    "to_2d_format",
    "to_3d_format",
    "test_for_normality",
]


def to_2d_format(three_dimensional_cov_matrices_array: np.array) -> np.array:
    """
    Convert estimates to 2D array to save it to disk.
    """
    return three_dimensional_cov_matrices_array.reshape(
        len(three_dimensional_cov_matrices_array), -1
    ).T  # (D*D, N)


def to_3d_format(r_formatted_array: np.array) -> np.array:
    """
    We cannot store a 3D object in a `.csv` file, so it is stored as a 2D matrix.
    This function reshapes it to a 3D object.

    Parameters
    ----------
    :param r_formatted_array: array of shape (D*D, N).
    :return: array of shape (N, D, D).
    """
    n_time_series = int(np.sqrt(r_formatted_array.shape[0]))
    assert len(r_formatted_array.shape) == 2
    three_dimensional_cov_matrices_array = np.reshape(
        r_formatted_array,
        (n_time_series, n_time_series, r_formatted_array.shape[1])
    )
    three_dimensional_cov_matrices_array = np.transpose(three_dimensional_cov_matrices_array, (2, 1, 0))
    return three_dimensional_cov_matrices_array


def test_for_normality(data_array: np.array) -> None:
    """
    Test for normality - a precondition for running t-tests.
    """
    k2, cohort_normality_pvalues = scipy.stats.normaltest(
        data_array, axis=0, nan_policy='omit'
    )
    print(k2.shape, cohort_normality_pvalues.shape)
