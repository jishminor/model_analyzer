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

from .server import AdmissionControllerServer
from model_analyzer.constants import SERVER_OUTPUT_TIMEOUT_SECS
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from subprocess import Popen, DEVNULL, STDOUT, TimeoutExpired
import psutil
import logging
import os

logger = logging.getLogger(__name__)


class AdmissionControllerServerLocal(AdmissionControllerServer):
    """
    Concrete Implementation of AdmissionControllerServer interface that runs
    admission_controller server locally as as subprocess.
    """

    def __init__(self, path,log_path):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the admission_controller server executable
        log_path: str
            Absolute path to the triton log file
        """

        self._admission_controller_process = None
        self._server_path = path
        self._log_path = log_path

    def start(self):
        """
        Starts the admission_controller server process locally
        """

        if self._server_path:
            # Create admission_controller config and run subprocess
            cmd = [self._server_path, 'main:app', '--port 5000']
            
            if self._log_path:
                try:
                    self._log_file = open(self._log_path, 'a+')
                except OSError as e:
                    raise TritonModelAnalyzerException(e)
            else:
                self._log_file = DEVNULL
            self._admission_controller_process = Popen(cmd,
                                               stdout=self._log_file,
                                               stderr=STDOUT,
                                               start_new_session=True,
                                               universal_newlines=True)

            logger.info('Admission Controller Server started.')

    def stop(self):
        """
        Stops the running admission_controller server
        """

        # Terminate process, capture output
        if self._admission_controller_process is not None:
            self._admission_controller_process.terminate()
            try:
                self._admission_controller_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS)
            except TimeoutExpired:
                self._admission_controller_process.kill()
                self._admission_controller_process.communicate()
            self._admission_controller_process = None
            if self._log_path:
                self._log_file.close()
            logger.info('Admission Controller Server stopped.')

    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

        if self._admission_controller_process:
            server_process = psutil.Process(self._admission_controller_process.pid)
            process_memory_info = server_process.memory_full_info()
            system_memory_info = psutil.virtual_memory()

            # Divide by 1.0e6 to convert from bytes to MB
            return (process_memory_info.uss //
                    1.0e6), (system_memory_info.available // 1.0e6)
        else:
            return 0.0, 0.0
