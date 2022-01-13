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

from .server_local import AdmissionControllerServerLocal
from .server_docker import AdmissionControllerServerDocker


class AdmissionControllerServerFactory:
    """
    A factory for creating TritonServer instances
    """

    @staticmethod
    def create_server_local(path, log_path=None):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the uvicorn executable
        log_path: str
            Absolute path to the fastapi log file

        Returns
        -------
        AdmissionControllerServerLocal
        """

        return AdmissionControllerServerLocal(
            path=path,
            log_path=log_path)

    @staticmethod
    def create_server_docker(image, config, log_path=None):
        """
        Parameters
        ----------
        image : str
            The fastapi docker image to pull and run
        log_path: str
            Absolute path to the triton log file

        Returns
        -------
        AdmissionControllerServerDocker
        """

        return AdmissionControllerServerDocker(
            image=image,
            log_path=log_path)
