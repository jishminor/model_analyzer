# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .common import test_result_collector as trc

from .mocks.mock_config import MockConfig
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_run_configs import MockRunConfigs

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.config.run.run_config_generator import RunConfigGenerator
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_manager import ModelManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from unittest.mock import MagicMock
from unittest.mock import patch


class ModelManagerSubclass(ModelManager):
    """ 
    Overrides execute_run_configs() to gather a list of MockRunConfigs that
    contain the configured values of each would-be 'executed' run_config
    """

    def __init__(self, config, client, server, metrics_manager, result_manager,
                 state_manager):
        super().__init__(config, client, server, metrics_manager,
                         result_manager, state_manager)
        self._configs = MockRunConfigs()

    def _execute_run_configs(self):
        while self._run_config_generator.run_configs():
            run_config = self._run_config_generator.next_config()
            self._configs.add_from_run_config(run_config)

    def get_run_configs(self):
        return self._configs


@patch('model_analyzer.config.run.run_search.RunSearch.add_measurements',
       MagicMock())
class TestModelManager(trc.TestResultCollector):

    def __init__(self, methodname):
        super().__init__(methodname)
        self._args = [
            'model-analyzer', 'profile', '--model-repository', 'cli_repository',
            '-f', 'path-to-config-file'
        ]

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

    def test_full_sweep(self):
        """
        Test a normal full sweep of options
        """
        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5],
            'kind': ["KIND_GPU"],
            'batching': [None, 0, 1, 2, 4, 8, 16],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 5
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_another_full_sweep(self):
        """
        Test another full sweep of options
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [None, 0, 1, 2, 4, 8],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_preferred_batch_size_disable(self):
        """
        Test with search_preferred_batch_size_disable=True
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8, 16, 32]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : True
            run_config_search_disable: False
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_run_search_disable(self):
        """
        Test with run_config_search_disable=True

        Expect 1 result that matches the default configuration because no manual 
        search options provided and automatic search disabled/ignored
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: True
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_manual_concurrency(self):
        """
        Test with manually specified concurrencies
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [None, 0, 1, 2, 4, 8],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [5, 7]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [5, 7]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            concurrency: [5, 7]
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_remote_mode(self):
        """
        Test remote mode

        In remote mode all model_config_parameters (preferred_batch_size, instance count) are ignored
        """

        expected_ranges = [{
            'instances': [None],
            'kind': [None],
            'batching': [None],
            'batch_sizes': [1],
            'concurrency': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 512
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            triton_launch_mode: remote            
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_manual_parameters(self):
        """
        Test with manually specified concurrencies and batch sizes
        """

        expected_ranges = [{
            'instances': [1, 2, 3, 4, 5, 6, 7],
            'kind': ["KIND_GPU"],
            'batching': [None, 0, 1, 2, 4, 8],
            'batch_sizes': [1, 2, 3],
            'max_batch_size': [8],
            'concurrency': [2, 10, 18, 26, 34, 42, 50, 58]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1, 2, 3],
            'max_batch_size': [8],
            'concurrency': [2, 10, 18, 26, 34, 42, 50, 58]
        }]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_concurrency: 512
            run_config_search_max_preferred_batch_size: 8
            run_config_search_max_instance_count: 7
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            concurrency:
                start: 2
                stop: 64
                step: 8
            batch_sizes: 1,2,3     
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_triton_parameters(self):
        """
        Test with manually specified triton options. 
        
        In this case we don't automatically search instances or dynamic_batching
        since model config parameters are specified.
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batch_sizes': [1],
            'max_batch_size': [1, 2, 4, 8, 16],
            'concurrency': [1, 2, 4, 8]
        }]

        yaml_content = """
            run_config_search_max_concurrency: 8
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 16
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8,16]
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_no_dynamic_batching_off(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (1 instance, max_batch_size=8, dynamic batching off)
        We should have a case of dynamic_batching off even though manual search only has it on
        """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [1, 2, 3],
            'max_queue_delay': ['200', '300'],
            'batch_sizes': [1],
            'max_batch_size': [1, 2, 4, 8, 16],
            'concurrency': [1, 2, 4, 8]
        }, {
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4, 8]
        }]

        yaml_content = """
            run_config_search_max_concurrency: 8
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 16
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8,16]
                        dynamic_batching:
                            preferred_batch_size: [[1], [2], [3]]
                            max_queue_delay_microseconds: [200, 300]                        
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_wrong_instances(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (2 instances, max_batch_size=8, dynamic batching off)
        We should have a 2-instance case even though manual search only has 1-instance
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 2
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [2],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = """
            run_config_search_max_concurrency: 4
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 16
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_cpu_vs_gpu(self):
        """
        If the default configuration had KIND_CPU, make sure it is run (even if everything
        else is the same)
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = """
            run_config_search_max_concurrency: 4
            run_config_search_max_preferred_batch_size: 16
            run_config_search_max_instance_count: 16
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """

        self._test_model_manager(yaml_content, expected_ranges)

    def test_default_config_always_run_automatic_search(self):
        """
        Test that the default config is run even when automatic search excludes that case
        In this case, default config is (4 instance, CPU, max_batch_size=8, dynamic batching off)
        We should have this 4 instance case though run_config_search_max_instance_count=1
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 4
            }
            ]
            """

        expected_ranges = [{
            'instances': [1],
            'kind': ["KIND_GPU"],
            'batching': [None, 0, 1, 2],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }, {
            'instances': [4],
            'kind': ["KIND_CPU"],
            'batching': [None],
            'batch_sizes': [1],
            'max_batch_size': [8],
            'concurrency': [1, 2, 4]
        }]

        yaml_content = """
            run_config_search_max_concurrency: 4
            run_config_search_max_preferred_batch_size: 2
            run_config_search_max_instance_count: 1
            run_config_search_preferred_batch_size_disable : False
            run_config_search_disable: False
            profile_models: test_model
            """
        self._test_model_manager(yaml_content, expected_ranges)

    def _test_model_manager(self, yaml_content, expected_ranges):
        """ 
        Test helper function that passes the given yaml_content into
        model_manager, runs the model, and confirms the result is as expected
        based on a full cartesian product of the lists in the input list of 
        dicts expected_ranges
        """

        # Use mock model config or else TritonModelAnalyzerException will be thrown as it tries to read from disk
        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = self._evaluate_config(self._args, yaml_content)

        state_manager = AnalyzerStateManager(config, MagicMock())
        model_manager = ModelManagerSubclass(config, MagicMock(), MagicMock(),
                                             MagicMock(), MagicMock(),
                                             state_manager)

        model_manager.run_model(config.profile_models[0])
        self.mock_model_config.stop()

        self._check_results(model_manager, expected_ranges)

    def _check_results(self, model_manager, expected_ranges):
        """ 
        Create a set of expected and actual run configs and confirm they are equal
        """
        run_configs = model_manager.get_run_configs()
        expected_configs = MockRunConfigs()
        expected_configs.populate_from_ranges(expected_ranges)

        self.assertEqual(run_configs.get_configs_set(),
                         expected_configs.get_configs_set())

    def _evaluate_config(self, args, yaml_content):
        """ Parse the given yaml_content into a config and return it """

        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(
            cmd='profile',
            help=
            'Run model inference profiling based on specified CLI or config options.',
            config=config)
        cli.parse()
        mock_config.stop()
        return config