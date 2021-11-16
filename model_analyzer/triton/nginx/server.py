# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod


class NginxServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """
    @abstractmethod
    def start(self):
        """
        Starts the tritonserver
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    @abstractmethod
    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

    def update_config(self, model_constraints):
        """
        Update the nginx config file

        Parameters
        ----------
        model_constraints: dict
            keys are argument names and values are their values.
        """

        self._nginx_config.set_model_constraints(model_constraints)
        self._nginx_config.update_nginx_config()