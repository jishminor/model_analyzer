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

import docker
import logging
import os
from multiprocessing.pool import ThreadPool

from .server import AdmissionControllerServer
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

LOCAL_HTTP_PORT = 8000
LOCAL_GRPC_PORT = 8001

logger = logging.getLogger(__name__)


class AdmissionControllerServerDocker(AdmissionControllerServer):
    """
    Concrete Implementation of AdmissionControllerServer interface that runs
    admission_controller in a docker container.
    """

    def __init__(self, image, config, log_path):
        """
        Parameters
        ----------
        image : str
            The admission_controller docker image to pull and run
        config : AdmissionControllerServerConfig
            the config object containing arguments for this server instance
        log_path: str
            Absolute path to the admission_controller log file
        """

        self._admission_controller_config = config
        self._docker_client = docker.from_env()
        self._admission_controller_image = image
        self._admission_controller_container = None
        self._log_path = log_path


    def start(self):
        """
        Starts the admission_controller docker container using docker-py
        """

        # Generate the admission_controller config to mount in the container
        with open(self._admission_controller_config.get_config_path(), 'w+') as output_file_:
            output_file_.write(self._admission_controller_config.get_admission_controller_config())

        # Mount required directories
        dirpath = os.path.dirname(self._admission_controller_config.get_config_path())
        volumes = {
            dirpath: {
                'bind': dirpath,
                'mode': 'ro'
            }
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        admission_controller_http_port = self._admission_controller_config.get_analyzer_config()['admission_controller_http_endpoint'].split(':')[-1]
        admission_controller_grpc_port = self._admission_controller_config.get_analyzer_config()['admission_controller_grpc_endpoint'].split(':')[-1]
        server_http_port = self._admission_controller_config.get_analyzer_config()['triton_http_endpoint'].split(':')[-1]
        server_grpc_port = self._admission_controller_config.get_analyzer_config()['triton_grpc_endpoint'].split(':')[-1]

        ports = {
            admission_controller_http_port: admission_controller_http_port,
            admission_controller_grpc_port: admission_controller_grpc_port,
            server_http_port: server_http_port,
            server_grpc_port: server_grpc_port,
        }

        try:
            # Run the docker container and run the command in the container
            self._admission_controller_container = self._docker_client.containers.run(
                command='admission_controller -c ' + self._admission_controller_config.get_config_path(),
                name='admission_controller',
                image=self._admission_controller_image,
                volumes=volumes,
                ports=ports,
                publish_all_ports=True,
                tty=False,
                stdin_open=False,
                detach=True)
        except docker.errors.APIError as error:
            if error.explanation.find('port is already allocated') != -1:
                raise TritonModelAnalyzerException(
                    "One of the following port(s) are already allocated: "
                    f"{admission_controller_http_port}, {admission_controller_grpc_port}, {server_http_port}, {server_grpc_port} "
                    "Change the AdmissionController server ports using"
                    " --admission_controller-http-endpoint, --admission_controller-grpc-endpoint")
            else:
                raise error

        if self._log_path:
            try:
                self._log_file = open(self._log_path, 'a+')
                self._log_pool = ThreadPool(processes=1)
                self._log_pool.apply_async(self._logging_worker)
            except OSError as e:
                raise TritonModelAnalyzerException(e)

        logger.info('Docker AdmissionController Container started.')


    def _logging_worker(self):
        """
        streams logs to
        log file
        """

        for chunk in self._admission_controller_container.logs(stream=True):
            self._log_file.write(chunk.decode('utf-8'))

    def stop(self):
        """
        Stops the admission_controller docker container
        and cleans up docker client
        """

        logger.info('Stopping admission_controller server container.')

        if self._admission_controller_container is not None:
            if self._log_path:
                if self._log_pool:
                    self._log_pool.terminate()
                    self._log_pool.close()
                if self._log_file:
                    self._log_file.close()
            self._admission_controller_container.stop()
            self._admission_controller_container.remove(force=True)
            self._admission_controller_container = None
            self._docker_client.close()


    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

        cmd = 'bash -c "pmap -x $(pgrep admission_controller) | tail -n1 | awk \'{print $4}\'"'
        _, used_mem_bytes = self._admission_controller_container.exec_run(cmd=cmd,
                                                                  stream=False)
        cmd = 'bash -c "free | awk \'{if(NR==2)print $7}\'"'
        _, available_mem_bytes = self._admission_controller_container.exec_run(
            cmd=cmd, stream=False)

        # Divide by 1.0e6 to convert from kilobytes to MB
        return float(used_mem_bytes.decode("utf-8")) // 1.0e3, float(
            available_mem_bytes.decode("utf-8")) // 1.0e3
