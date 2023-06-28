import numpy as np

__all__ = ["to_correlation_structure", "_correlation_from_covariance"]


def to_correlation_structure(covariance_structure: np.array) -> np.array:
    """
    Converts a covariance structure into a correlation structure.
    :param covariance_structure: array of shape (N, D, D).
    :return: array of shape (N, D, D)
    """
    correlation_structure = [
        _correlation_from_covariance(time_step_covariance_matrix) for time_step_covariance_matrix in covariance_structure
    ]
    return np.array(correlation_structure)


def _correlation_from_covariance(covariance_matrix: np.array) -> np.array:
    """
    Converts covariance matrix into a correlation matrix.
    TODO: perhaps we could merge this with nilearn.connectome.cov_to_corr
    """
    v = np.sqrt(np.diag(covariance_matrix))  # (D, )
    outer_v = np.outer(v, v)  # (D, D)
    correlation_matrix = covariance_matrix / outer_v  # (D, D)
    correlation_matrix[covariance_matrix == 0] = 0
    return correlation_matrix
