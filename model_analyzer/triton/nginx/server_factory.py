# Copyright (c) 2020,21 NVIDIA CORPORATION. All rights reserved.
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

from .server_local import NginxServerLocal
from .server_docker import NginxServerDocker


class NginxServerFactory:
    """
    A factory for creating TritonServer instances
    """

    @staticmethod
    def create_server_local(path, config, log_path=None):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : NginxServerConfig
            the config object containing arguments for this server instance
        log_path: str
            Absolute path to the triton log file

        Returns
        -------
        NginxServerLocal
        """

        return NginxServerLocal(
            path=path,
            config=config,
            log_path=log_path)

    @staticmethod
    def create_server_docker(image, config, log_path=None):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        log_path: str
            Absolute path to the triton log file

        Returns
        -------
        NginxServerDocker
        """

        return NginxServerDocker(
            image=image,
            config=config,
            log_path=log_path)
