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
    DEFAULT_CHECKPOINT_DIRECTORY, DEFAULT_DURATION_SECONDS, DEFAULT_RUN_CONFIG_MAX_CONCURRENCY, \
    DEFAULT_GPUS, DEFAULT_CB_SEARCH_ITERATIONS, DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, \
    DEFAULT_MONITORING_INTERVAL, DEFAULT_OFFLINE_OBJECTIVES, DEFAULT_CB_SEARCH_ADF, \
    DEFAULT_CB_SEARCH_NO_LEARN, DEFAULT_CB_SEARCH_EPSILON, DEFAULT_CB_SEARCH_CONTEXTS, \
    DEFAULT_RUN_CONFIG_MAX_PREFERRED_BATCH_SIZE, DEFAULT_RUN_CONFIG_PREFERRED_BATCH_SIZE_DISABLE, \
    DEFAULT_CB_SEARCH_EXPLORATION, DEFAULT_CB_SEARCH_TAU, DEFAULT_CB_SEARCH_POLICIES, \
    DEFAULT_CB_SEARCH_LAMBDA, DEFAULT_CB_SEARCH_RND, DEFAULT_CB_SEARCH_ENCODE_CONTEXT_NUMERIC, \
    DEFAULT_CB_SEARCH_ENCODE_ACTIONS_NUMERIC

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
        self._add_config(
            ConfigField(
                'run_config_search_max_concurrency',
                flags=['--run-config-search-max-concurrency'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_CONCURRENCY,
                description=
                "Max concurrency value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_instance_count',
                flags=['--run-config-search-max-instance-count'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT,
                description=
                "Max instance count value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_max_preferred_batch_size',
                flags=['--run-config-search-max-preferred-batch-size'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_RUN_CONFIG_MAX_PREFERRED_BATCH_SIZE,
                description=
                "Max preferred batch size value that run config search should not go beyond that."
            ))
        self._add_config(
            ConfigField(
                'run_config_search_preferred_batch_size_disable',
                flags=['--run-config-search-preferred-batch-size-disable'],
                field_type=ConfigPrimitive(bool),
                default_value=DEFAULT_RUN_CONFIG_PREFERRED_BATCH_SIZE_DISABLE,
                description="Disable preferred batch size search."))

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
                'cpu_used_ram':
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
                'profile_models',
                flags=['--profile-models'],
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
                'encode_context_numeric',
                flags=['--encode-context-numeric'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_CB_SEARCH_ENCODE_CONTEXT_NUMERIC,
                description="Enable encoding of context values as numeric"))
        self._add_config(
            ConfigField(
                'encode_actions_numeric',
                flags=['--encode-actions-numeric'],
                field_type=ConfigPrimitive(bool),
                parser_args={'action': 'store_true'},
                default_value=DEFAULT_CB_SEARCH_ENCODE_ACTIONS_NUMERIC,
                description="Enable encoding of action values as numeric"))
        self._add_config(
            ConfigField(
                'iterations',
                flags=['--iterations'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_ITERATIONS,
                description="Number of iterations for CB search"))
        self._add_config(
            ConfigField(
                'exploration',
                flags=['--exploration'],
                choices=['epsilon', 'first', 'bag', 'cover', 'softmax', 'rnd'],
                field_type=ConfigPrimitive(str),
                default_value=DEFAULT_CB_SEARCH_EXPLORATION,
                description=
                'Exploration method for cb search (cover only available if adf disabled)'
            ))
        self._add_config(
            ConfigField(
                'epsilon',
                flags=['--epsilon'],
                field_type=ConfigPrimitive(float),
                default_value=DEFAULT_CB_SEARCH_EPSILON,
                description="Epsilon value for CB search"))
        self._add_config(
            ConfigField(
                'tau',
                flags=['--tau'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_TAU,
                description="Tau parameter for first search"))
        self._add_config(
            ConfigField(
                'policies',
                flags=['--policies'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_POLICIES,
                description="Num policy parameter for bag and cover search"))
        self._add_config(
            ConfigField(
                'lambda_value',
                flags=['--lambda'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_LAMBDA,
                description="Lambda value for softmax search"))
        self._add_config(
            ConfigField(
                'rnd',
                flags=['--rnd'],
                field_type=ConfigPrimitive(int),
                default_value=DEFAULT_CB_SEARCH_RND,
                description="RND value for CB search using rnd"))
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
                default_value=DEFAULT_CB_SEARCH_CONTEXTS,
                description=
                'Contexts to use for CB learning'
            ))

    def _preprocess_and_verify_arguments(self):
        """
        Enforces some rules on the config.

        Raises
        ------
        TritonModelAnalyzerException
            If there is a problem with arguments or config.
        """

        if self.triton_launch_mode == 'remote':
            if self.client_protocol == 'http' and not self.triton_http_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'http'. Must specify triton-http-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")
            if self.client_protocol == 'grpc' and not self.triton_grpc_endpoint:
                raise TritonModelAnalyzerException(
                    "client-protocol is 'grpc'. Must specify triton-grpc-endpoint "
                    "if connecting to already running server or change protocol using "
                    "--client-protocol.")

    def _autofill_values(self):
        """
        Fill in the implied or default
        config values.
        """

        cpu_only = False
        if not numba.cuda.is_available():
            cpu_only = True

        new_profile_models = {}
        for model in self.profile_models:
            new_model = {'cpu_only': (model.cpu_only() or cpu_only)}

            # Objectives
            if not model.objectives():
                new_model['objectives'] = self.objectives
            else:
                new_model['objectives'] = model.objectives()

            # Constraints
            if not model.constraints():
                if 'constraints' in self._fields and self._fields[
                        'constraints'].value():
                    new_model['constraints'] = self.constraints
            else:
                new_model['constraints'] = model.constraints()

            # Perf analyzer flags
            if not model.perf_analyzer_flags():
                new_model['perf_analyzer_flags'] = self.perf_analyzer_flags
            else:
                new_model['perf_analyzer_flags'] = model.perf_analyzer_flags()

            # Perf analyzer flags
            if not model.triton_server_flags():
                new_model['triton_server_flags'] = self.triton_server_flags
            else:
                new_model['triton_server_flags'] = model.triton_server_flags()

            # Transfer model config parameters directly
            if model.model_config_parameters():
                new_model[
                    'model_config_parameters'] = model.model_config_parameters(
                    )

            new_profile_models[model.model_name()] = new_model
        self._fields['profile_models'].set_value(new_profile_models)