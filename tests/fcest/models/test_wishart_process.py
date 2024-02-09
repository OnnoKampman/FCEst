import logging
import unittest

import gpflow
from gpflow.ci_utils import ci_niter
import numpy as np

from fcest.helpers.inference import run_adam_svwp, run_adam_vwp
from fcest.models.wishart_process import SparseVariationalWishartProcess, VariationalWishartProcess

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestWishartProcess(unittest.TestCase):

    @staticmethod
    def _generate_dummy_data() -> (np.array, np.array):

        n_time_series = 2
        n_time_steps = 7

        x = np.linspace(0, 1, n_time_steps).reshape(-1, 1)
        y = np.random.random(size=(n_time_steps, n_time_series))

        return x, y

    def test_sparse_variational_wishart_process(self):
        """
        Test instantiation of SparseVariationalWishartProcess.
        """
        x, y = self._generate_dummy_data()  # (N, 1), (N, D)

        k = gpflow.kernels.Matern52()
        m = SparseVariationalWishartProcess(
            D=y.shape[1],
            Z=np.arange(y.shape[0]),
            nu=y.shape[1],
            kernel=k,
        )
        maxiter = ci_niter(3)
        logf = run_adam_svwp(
            m,
            data=(x, y),
            iterations=maxiter,
            log_interval=100,
            log_dir=None,
        )

    def test_variational_wishart_process(self):
        """
        Test instantiation of VariationalWishartProcess.
        """
        x, y = self._generate_dummy_data()  # (N, 1), (N, D)

        k = gpflow.kernels.Matern52()
        m = VariationalWishartProcess(
            x_observed=x,
            y_observed=y,
            nu=y.shape[1],
            kernel=k,
        )
        maxiter = ci_niter(3)
        logf = run_adam_vwp(
            m,
            iterations=maxiter,
            log_interval=100,
            log_dir=None,
        )


if __name__ == "__main__":
    unittest.main()
