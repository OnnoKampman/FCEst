import logging

import numpy as np
import tensorflow as tf

__all__ = [
    "to_correlation_structure", 
    "_correlation_from_covariance",
    "convert_tensor_to_correlation",
    "are_all_positive_definite",
]


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


def convert_tensor_to_correlation(covariance_matrix: tf.Tensor) -> tf.Tensor:
    """
    Convert covariance tensor to correlation tensor.
    :param covariance_matrix:
    :return:
    """
    diag_covs = tf.linalg.diag_part(covariance_matrix)
    diag_covs = tf.expand_dims(diag_covs, axis=-1)
    diag_covs_outer = tf.matmul(diag_covs, diag_covs, transpose_b=True)
    diag_covs_outer = tf.math.sqrt(diag_covs_outer)
    correlation_matrix = covariance_matrix / diag_covs_outer
    return correlation_matrix


def are_all_positive_definite(covariance_matrices: tf.Tensor) -> bool:
    """Checks if collection of covariance matrices are all positive definite."""
    for covariance_matrix_tensor in covariance_matrices:
        # check covariance matrix predictions are symmetric
        if not np.allclose(
            (covariance_matrix_tensor.numpy() - covariance_matrix_tensor.numpy().T),
            0,
            rtol=1e-6
        ):
            logging.warning('Matrix is not symmetric.')
            print(covariance_matrix_tensor)
            return False
        # check if positive definite
        try:
            _ = np.linalg.cholesky(covariance_matrix_tensor)
            continue
        except np.linalg.LinAlgError as e_linalg:
            print(e_linalg)
            return False
    return True
