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

import numba
import psutil
from .config_field import ConfigField
from .config_primitive import ConfigPrimitive
from .config_list_string import ConfigListString
from .config_list_numeric import ConfigListNumeric
from .config_object import ConfigObject
from .config_list_generic import ConfigListGeneric
from .config_union import ConfigUnion
from .config_none import ConfigNone
from .config_sweep import ConfigSweep
from .config_enum import ConfigEnum
from .config_command import ConfigCommand
from .config_command_profile import ConfigCommandProfile

from .config_defaults import \
    DEFAULT_BATCH_SIZES, DEFAULT_CHECKPOINT_DIRECTORY, \
    DEFAULT_CLIENT_PROTOCOL, DEFAULT_DURATION_SECONDS, \
    DEFAULT_GPUS, DEFAULT_MAX_RETRIES, DEFAULT_CB_SEARCH_ITERATIONS, \
    DEFAULT_MONITORING_INTERVAL, DEFAULT_OFFLINE_OBJECTIVES, DEFAULT_CB_SEARCH_ADF, \
    DEFAULT_CB_SEARCH_NO_LEARN, DEFAULT_CB_CONTEXT_LIST, DEFAULT_CB_SEARCH_EPSILON, \
    DEFAULT_OUTPUT_MODEL_REPOSITORY, DEFAULT_OVERRIDE_OUTPUT_REPOSITORY_FLAG, \
    DEFAULT_PERF_ANALYZER_CPU_UTIL, DEFAULT_PERF_ANALYZER_PATH, DEFAULT_PERF_MAX_AUTO_ADJUSTS, \
    DEFAULT_PERF_OUTPUT_FLAG, DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, \
    DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, DEFAULT_RUN_CONFIG_MAX_PREFERRED_BATCH_SIZE, \
    DEFAULT_RUN_CONFIG_SEARCH_DISABLE, DEFAULT_TRITON_DOCKER_IMAGE, DEFAULT_TRITON_GRPC_ENDPOINT, \
    DEFAULT_TRITON_HTTP_ENDPOINT, DEFAULT_TRITON_LAUNCH_MODE, DEFAULT_TRITON_METRICS_URL, \
    DEFAULT_TRITON_SERVER_PATH, DEFAULT_PERF_ANALYZER_TIMEOUT

from .objects.config_model_profile_spec import ConfigModelProfileSpec
from model_analyzer.triton.server.server_config import \
    TritonServerConfig
from model_analyzer.perf_analyzer.perf_config import \
    PerfAnalyzerConfig
from model_analyzer.record.record import RecordType
from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException
from .objects.config_protobuf_utils import \
    is_protobuf_type_primitive, protobuf_to_config_type

from tritonclient.grpc.model_config_pb2 import ModelConfig
from google.protobuf.descriptor import FieldDescriptor


class ConfigCommandCBSearch(ConfigCommandProfile):
    def _fill_config(self):
        """
        Builder function makes calls to add config to 
        fill the config with options
        """

        self._add_config(
            ConfigField(
                'config_file',
                field_type=ConfigPrimitive(str),
                flags=['-f', '--config-file'],
                description="Path to Config File for subcommand 'profile'."))
        self._add_config(
            ConfigField(
                'checkpoint_directory',
                flags=['--checkpoint-directory', '-s'],
                default_value=DEFAULT_CHECKPOINT_DIRECTORY,
                field_type=ConfigPrimitive(str),
                description=
                "Full path to directory to which to read and write checkpoints and profile data."
            ))
        self._add_config(
            ConfigField(
                'monitoring_interval',
                flags=['-i', '--monitoring-interval'],
                field_type=ConfigPrimitive(float),
                default_value=DEFAULT_MONITORING_INTERVAL,
                description=
                'Interval of time between DGCM measurements in seconds'))
        self._add_config(
            ConfigField(
                'duration_seconds',
                field_type=ConfigPrimitive(int),
                flags=['-d', '--duration-seconds'],
                default_value=DEFAULT_DURATION_SECONDS,
                description=
                'Specifies how long (seconds) to gather server-only metrics'))
        self._add_config(
            ConfigField(
                'gpus',
                flags=['--gpus'],
                field_type=ConfigListString(),
                default_value=DEFAULT_GPUS,
                description="List of GPU UUIDs to be used for the profiling. "
                "Use 'all' to profile all the GPUs visible by CUDA."))

        self._add_repository_configs()
        self._add_client_configs()
        self._add_cb_search_models_configs()
        self._add_perf_analyzer_configs()
        self._add_triton_configs()

    def _add_cb_search_models_configs(self):
        """
        Adds configs specific to model specifications
        """
        triton_server_flags_scheme = ConfigObject(schema={
            k: ConfigPrimitive(str)
            for k in TritonServerConfig.allowed_keys()
        })
        perf_analyzer_flags_scheme = ConfigObject(
            schema={
                k: ConfigPrimitive(type_=str)
                for k in PerfAnalyzerConfig.allowed_keys()
            })
        self._add_config(
            ConfigField(
                'perf_analyzer_flags',
                field_type=perf_analyzer_flags_scheme,
                description=
                'Allows custom configuration of the perf analyzer instances used by model analyzer.'
            ))
        self._add_config(
            ConfigField(
                'triton_server_flags',
                field_type=triton_server_flags_scheme,
                description=
                'Allows custom configuration of the triton instances used by model analyzer.'
            ))

        def objective_list_output_mapper(objectives):
            # Takes a list of objectives and maps them
            # into a dict
            output_dict = {}
            for objective in objectives:
                value = ConfigPrimitive(type_=int)
                value.set_value(10)
                output_dict[objective] = value
            return output_dict

        objectives_scheme = ConfigUnion([
            ConfigObject(
                schema={
                    tag: ConfigPrimitive(type_=int)
                    for tag in RecordType.get_all_record_types().keys()
                }),
            ConfigListString(output_mapper=objective_list_output_mapper)
        ])
        constraints_scheme = ConfigObject(
            schema={
                'perf_throughput':
                ConfigObject(schema={
                    'min': ConfigPrimitive(int),
                }),
                'perf_latency':
                ConfigObject(schema={
                    'max': ConfigPrimitive(int),
                }),
                'gpu_used_memory':
                ConfigObject(schema={
                    'max': ConfigPrimitive(int),
                }),
            })
        self._add_config(
            ConfigField(
                'objectives',
                field_type=objectives_scheme,
                default_value=DEFAULT_OFFLINE_OBJECTIVES,
                description=
                'Model Analyzer uses the objectives described here to find the best configuration for each model.'
            ))
        self._add_config(
            ConfigField(
                'constraints',
                field_type=constraints_scheme,
                description=
                'Constraints on the objectives specified in the "objectives" field of the config.'
            ))
        model_config_fields = self._get_model_config_fields()
        profile_model_scheme = ConfigObject(
            required=True,
            schema={
                # Any key is allowed, but the keys must follow the pattern
                # below
                '*':
                ConfigObject(
                    schema={
                        'cpu_only':
                        ConfigPrimitive(bool),
                        'parameters':
                        ConfigObject(
                            schema={
                                'batch_sizes': ConfigListNumeric(type_=int),
                                'concurrency': ConfigListNumeric(type_=int)
                            }),
                        'objectives':
                        objectives_scheme,
                        'constraints':
                        constraints_scheme,
                        'model_config_parameters':
                        model_config_fields,
                        'perf_analyzer_flags':
                        perf_analyzer_flags_scheme,
                        'triton_server_flags':
                        triton_server_flags_scheme,
                    })
            },
            output_mapper=ConfigModelProfileSpec.
            model_object_to_config_model_profile_spec)
        self._add_config(
            ConfigField(
                'model_set',
                flags=['--model-set'],
                field_type=ConfigUnion([
                    profile_model_scheme,
                    ConfigListGeneric(
                        ConfigUnion([
                            profile_model_scheme,
                            ConfigPrimitive(
                                str,
                                output_mapper=ConfigModelProfileSpec.
                                model_str_to_config_model_profile_spec)
                        ]),
                        required=True,
                        output_mapper=ConfigModelProfileSpec.
                        model_mixed_to_config_model_profile_spec),
                    ConfigListString(output_mapper=ConfigModelProfileSpec.
                                     model_list_to_config_model_profile_spec),
                ],
                                       required=True),
                description='List of the models to be profiled'))
        self._add_config(
            ConfigField(
                'adf',
                flags=['--adf'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_CB_SEARCH_ADF,
                description="Enable action dependent features for VW"))
        self._add_config(
            ConfigField(
                'iterations',
                flags=['--iterations'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_ITERATIONS,
                description="Number of iterations for CB search"))
        self._add_config(
            ConfigField(
                'epsilon',
                flags=['--epsilon'],
                field_type=ConfigPrimitive(float),
                default_value=DEFAULT_CB_SEARCH_EPSILON,
                description="Epsilon value for CB search"))
        self._add_config(
            ConfigField(
                'no_learn',
                flags=['--no-learn'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_CB_SEARCH_NO_LEARN,
                description="Disable CB learning during"))
        self._add_config(
            ConfigField(
                'context_list',
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_CB_CONTEXT_LIST,
                flags=['--context-list'],
                description=
                'file holding list of context dictionaries to be used'
            ))
        self._add_config(
            ConfigField(
                'contexts',
                flags=['--contexts'],
                choices=['concurrency-range', 'batch-size'],
                field_type=ConfigListString(str),
                description=
                'Contexts to use for CB learning'
            ))