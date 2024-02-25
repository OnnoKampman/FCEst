# Copyright 2020-2024 The FCEst Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from ..helpers.array_operations import to_correlation_structure
from ..helpers.data import to_3d_format


class MGARCH:
    """
    MGARCH base class.
    """

    def __init__(
            self,
            mgarch_type: str,
            uspec_mean_model: str = "c(1, 1)",
            uspec_variance_model: str = "c(1, 1)",
            dccspec_order: str = "c(1, 1)",
    ) -> None:
        """
        You will need to run `install.packages('rmgarch')` in your R console.

        References
            Inspired by https://quant.stackexchange.com/questions/20687/multivariate-garch-in-python

        Parameters
        ----------
        :param mgarch_type:
            Either 'DCC' or 'GO'.
        :param uspec_mean_model:
        :param uspec_variance_model:
        :param dccspec_order:
        """
        match mgarch_type:
            case 'DCC':
                self._define_r_code_model_dcc(
                    uspec_mean_model=uspec_mean_model,
                    uspec_variance_model=uspec_variance_model,
                    dccspec_order=dccspec_order
                )
            case 'GO':
                self._define_r_code_model_go()
            case _:
                raise NotImplementedError(f"MGARCH model type '{mgarch_type:s}' not recognized.")

    def _define_r_code_model_dcc(
            self, uspec_mean_model: str, uspec_variance_model: str, dccspec_order: str,
            spec_distribution: str = "mvt", fit_control: str = "FALSE"
    ) -> None:
        """
        Note that the double curly brackets are necessary for Python's f-strings.

        References
            For more inspiration: https://github.com/canlab/Lindquist_Dynamic_Correlation

        Parameters
        ----------
        :param uspec_mean_model:
            Note: this used to be c(1, 0).
        :param uspec_variance_model:
            Note: this was not here before.
        :param dccspec_order:
            Default is c(1, 1).
        :param spec_distribution:
            "mvt" or "mvnorm"
        :param fit_control:
            Determines whether standard errors are computed.
        """
        r_dcc_garch_code = f"""
            library('rugarch')
            library('rmgarch')
            function(r_time_series){{
                num_time_steps <- dim(r_time_series)[1]
                num_time_series <- dim(r_time_series)[2]
                uspec <- multispec(
                    replicate(
                        num_time_series,
                        ugarchspec(
                            mean.model = list(armaOrder = {uspec_mean_model:s}),
                            variance.model = list(garchOrder = {uspec_variance_model:s})
                        )
                    )
                )
                spec <- dccspec(
                    uspec,
                    dccOrder = {dccspec_order:s},
                    distribution = "{spec_distribution:s}"
                )
                fit <- dccfit(
                    spec,
                    data = r_time_series,
                    solver = c('hybrid', 'solnp'),
                    fit.control = list(eval.se = {fit_control:s})
                )
                cov_matrices <- rcov(fit)  # (D, D, N)
                list(fit, cov_matrices)  # these will be returned by this function
            }}
        """
        self.mgarch = ro.r(r_dcc_garch_code)

    def _define_r_code_model_go(
            self, gogarchspec_mean_model: str = "c(1, 1)"
    ) -> None:
        """
        Note that the double curly brackets are necessary for Python's f-strings.
        """
        r_go_garch_code = f"""
            library('rmgarch')
            function(r_time_series){{
                num_time_steps <- dim(r_time_series)[1]
                num_time_series <- dim(r_time_series)[2]
                spec <- gogarchspec(
                    mean.model = list(demean = "constant"),
                    variance.model = list(model = "sGARCH", garchOrder = {gogarchspec_mean_model:s}, submodel = NULL),
                    distribution.model = "manig",
                    ica = "fastica"
                )
                fit <- gogarchfit(
                    spec = spec,
                    data = r_time_series,
                    gfun = "tanh",
                    solver = 'hybrid'
                )
                cov_matrices <- rcov(fit)  # (D, D, N)
                list(fit, cov_matrices)  # these will be returned by this function
            }}
        """
        self.mgarch = ro.r(r_go_garch_code)

    def predict_corr(self) -> np.array:
        """
        Returns full TVFC correlation structure.
        """
        return self.train_location_covariance_structure

    def fit_model(
            self, training_data_df: pd.DataFrame, training_type: str = 'joint'
    ) -> None:
        """
        Note that these MGARCH implementations require at least 100 time points to train.

        Parameters
        ----------
        :param training_data_df:
            Expected of shape (N, D).
        :param training_type:
            'joint' or 'bivariate_loop'.
        """
        if len(training_data_df) < 100:
            logging.error("Data length too short to train MGARCH models on.")
            exit()
        match training_type:
            case 'joint' | 'multivariate':
                fit, self.train_location_covariance_structure = self._fit_model_joint(training_data_df)
                self.fit = fit
            case 'bivariate_loop' | 'pairwise':
                self.train_location_covariance_structure = self._fit_model_bivariate_loop(training_data_df)
            case _:
                raise NotImplementedError(f"Training type '{training_type:s}' not recognized.")

    def _fit_model_joint(self, training_data_df: pd.DataFrame):
        """
        Fit the MGARCH model to the training data jointly.

        Parameters
        ----------
        :param training_data_df:
        :return:
            fit: object with fit results overview and parameters
            train_location_covariance_structure: array of shape (N, D, D)
        """
        r_training_data_df = self._convert_to_r_df(training_data_df)  # R format DataFrame
        fit, train_location_covariance_structure = self.mgarch(r_training_data_df)  # <class 'rpy2.robjects.methods.RS4'>, <class 'rpy2.robjects.vectors.FloatArray'>
        train_location_covariance_structure = np.array(train_location_covariance_structure)  # (D, D, N)
        train_location_covariance_structure = np.transpose(
            train_location_covariance_structure, (2, 0, 1)
        )  # (N, D, D)
        return fit, train_location_covariance_structure

    def _fit_model_bivariate_loop(self, training_data_df: pd.DataFrame) -> np.array:
        """
        Here we loop over all edges in pairwise fashion.

        Parameters
        ----------
        :param training_data_df:
            Expected of shape (N, D).
        :return:
        """
        num_time_steps = training_data_df.shape[0]
        num_time_series = training_data_df.shape[1]

        # Break DataFrame up into bivariate pairs (i.e. edgewise).
        interaction_pairs = np.triu_indices(num_time_series, k=1)
        interaction_pairs = list(zip(*interaction_pairs))  # list of interaction pairs

        # Train each bivariate pair and construct full covariance structure.
        train_location_covariance_structure = np.zeros((num_time_steps, num_time_series, num_time_series))
        for i_interaction_pair, (ts_i, ts_j) in enumerate(interaction_pairs):
            print(f"Edge {i_interaction_pair+1:d}/{len(interaction_pairs):d}.")
            bivariate_pair_df = training_data_df.iloc[:, [ts_i, ts_j]]
            _, cov_struc = self._fit_model_joint(bivariate_pair_df)  # (N, 2, 2)

            # Add covariance term to full covariance structure.
            train_location_covariance_structure[:, ts_i, ts_j] = cov_struc[:, 0, 1]
            train_location_covariance_structure[:, ts_j, ts_i] = cov_struc[:, 0, 1]

            # Add (mean) variance terms to full covariance structure.
            train_location_covariance_structure[:, ts_i, ts_i] += cov_struc[:, 0, 0] / (num_time_series - 1)
            train_location_covariance_structure[:, ts_j, ts_j] += cov_struc[:, 1, 1] / (num_time_series - 1)

        return train_location_covariance_structure

    def save_tvfc_estimates(
            self, savedir: str, model_name: str,
            connectivity_metric: str = 'correlation'
    ) -> None:
        """
        Save TVFC estimates.

        TODO: write unit test for this

        Parameters
        ----------
        :param savedir:
        :param model_name:
        :param connectivity_metric:
        """
        tlcs = self.train_location_covariance_structure  # (N, D, D)
        if connectivity_metric == 'correlation':
            tlcs = to_correlation_structure(tlcs)
        tlcs_df = pd.DataFrame(tlcs.reshape(len(tlcs), -1).T)  # (D*D, N)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        tlcs_df.to_csv(
            os.path.join(savedir, model_name)
        )
        logging.info(f"Saved MGARCH model (train location) estimates to '{savedir:s}'.")

    @staticmethod
    def load_model_estimates(savedir: str, model_name: str) -> np.array:
        """
        Load model TVFC estimates.
        """
        train_loc_cov_structure = pd.read_csv(os.path.join(savedir, model_name))  # (D*D, N)
        train_loc_cov_structure = to_3d_format(train_loc_cov_structure)  # (N, D, D)
        return train_loc_cov_structure

    @staticmethod
    def _convert_to_r_df(python_df: pd.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_python_df = ro.conversion.py2rpy(python_df)
        return r_from_python_df
