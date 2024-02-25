import logging
import unittest

import gpflow
from gpflow.ci_utils import ci_niter
import numpy as np

from fcest.helpers.inference import run_adam
from fcest.models.wishart_process import SparseVariationalWishartProcess, VariationalWishartProcess

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestWishartProcess(unittest.TestCase):

    @staticmethod
    def _generate_dummy_data() -> (np.array, np.array):

        num_time_series = 2
        num_time_steps = 7

        x = np.linspace(0, 1, num_time_steps).reshape(-1, 1)
        y = np.random.random(size=(num_time_steps, num_time_series))

        return x, y

    def test_sparse_variational_wishart_process(
        self,
        num_iterations: int = 3,
    ) -> None:
        """
        Test instantiation of SparseVariationalWishartProcess.
        """
        x, y = self._generate_dummy_data()  # (N, 1), (N, D)

        k = gpflow.kernels.Matern52()
        m = SparseVariationalWishartProcess(
            D=y.shape[1],
            Z=np.arange(y.shape[0]).reshape(-1, 1),
            nu=y.shape[1],
            kernel=k,
        )

        maxiter = ci_niter(num_iterations)
        logf = run_adam(
            model_type="SVWP",
            model=m,
            data=(x, y),
            iterations=maxiter,
            log_interval=1,
            log_dir=None,
        )

        self.assertEqual(type(logf), list)
        self.assertEqual(len(logf), maxiter)

    def test_variational_wishart_process(
        self,
        num_iterations: int = 3,
    ) -> None:
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

        maxiter = ci_niter(num_iterations)
        logf = run_adam(
            model_type="VWP",
            model=m,
            iterations=maxiter,
            log_interval=1,
            log_dir=None,
        )

        self.assertEqual(type(logf), list)
        self.assertEqual(len(logf), maxiter)


if __name__ == "__main__":
    unittest.main()
