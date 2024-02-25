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

import numpy as np


class HiddenMarkovModel:
    """
    Main class for Hidden Markov Models (HMM) for time-varying functional connectivity (TVFC) estimation.
    """

    def __init__(
            self,
            x_train_locations: np.array,
            y_train_locations: np.array,
    ) -> None:
        """
        Main HMM functional connectivity class.

        Parameters
        ----------
        :param x_train_locations:
            Array of expected shape (num_time_steps, 1), i.e. (N, 1)
        :param y_train_locations:
            Array of expected shape (num_time_steps, num_time_series), i.e. (N, D)
        :return:
        """
        self.x_train = x_train_locations
        self.y_train = y_train_locations
        assert len(x_train_locations) == len(y_train_locations)

        self.num_time_steps = y_train_locations.shape[0]  # N
        self.num_time_series = y_train_locations.shape[1]  # D

        raise NotImplementedError
