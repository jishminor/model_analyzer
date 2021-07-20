# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from .run_search import RunSearch
from model_analyzer.config.input.objects.config_model import ConfigModel
import logging
from model_analyzer.constants import THROUGHPUT_GAIN
import copy


class RunSearchCB(RunSearch):
    """
    A class responsible for searching the config space using CB techniques.
    """
    