import numpy as np

__all__ = ["to_3d_format"]


def to_3d_format(r_formatted_array: np.array) -> np.array:
    """
    We cannot store a 3D object in a `.csv` file, so it is stored as a 2D matrix.
    This function reshapes it to a 3D object.
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
