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

from model_analyzer.constants import LOGGER_NAME
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.output.file_writer import FileWriter
from .model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.config.run.run_search import RunSearch
from model_analyzer.config.run.run_search_cb import RunSearchCB
from model_analyzer.config.run.run_config_generator \
    import RunConfigGenerator
from model_analyzer.triton.nginx.server_factory import NginxServerFactory
from model_analyzer.triton.nginx.server_config import NginxServerConfig


from model_analyzer.config.run.run_config import RunConfig


import os
from collections import defaultdict
import pickle
import random
import shutil
import logging

logger = logging.getLogger(LOGGER_NAME)


class ModelManager:
    """
    This class handles the search for, creation of, and execution of run configs.
    It also records the best results for each model.
    """

    def __init__(self, config, client, server, metrics_manager, result_manager,
                 state_manager, nginx=None, admission_controller=None):
        """
        Parameters
        ----------
        config:ConfigCommandProfile
            The config for the model analyzer
        client: TritonClient
            The client handle used to send requests to Triton
        server: TritonServer
            The server handle used to start and stop Triton instances
        metrics_manager: MetricsManager
            The object that handles launching perf analyzer instances and profiling.
        result_manager: ResultManager
            The object that handles storing and sorting the results from the perf analyzer
        state_manager: AnalyzerStateManager
            The object that handles serializing the state of the analyzer and saving.
        nginx : NginxServer
            Nginx Instance to proxy triton api requests
        admission_controller : AdmissionController
            Admission Controller Instance to handle model fit
        """

        self._config = config
        self._client = client
        self._server = server
        self._metrics_manager = metrics_manager
        self._result_manager = result_manager
        self._state_manager = state_manager
        self._run_search = RunSearch(config=config)
        self._last_config_variant = None
        self._run_config_generator = RunConfigGenerator(config=config,
                                                        client=self._client)
        self._nginx = nginx
        self._admission_controller = admission_controller

        # Generate the output model repository path folder.
        self._output_model_repo_path = config.output_model_repository_path

    def run_model(self, model):
        """
        Generates configs, runs inferences, gets
        measurements for a single model

        Parameters
        ----------
        model : ConfigModelProfileSpec
            The model being run
        """

        # Clear any configs from previous model run
        self._run_config_generator.clear_configs()

        # Save the global server config and update the server's config for this model run
        server_config_copy = self._server.config().copy()
        self._server.update_config(params=model.triton_server_flags())

        # Run model inferencing
        if self._config.run_config_search_disable:
            logger.info(
                f"Running manual config search for model: {model.model_name()}")
            self._run_model_no_search(model)
        else:
            logger.info(
                f"Running auto config search for model: {model.model_name()}")
            self._run_model_with_search(model)

        # Reset the server args to global config
        self._server.update_config(params=server_config_copy.server_args())

    def cb_search_models(self, models):
        """
        Runs CB search over list of models in config
        """

        # Create nginx conf directory if not exists
        try:
            os.mkdir(self._config.nginx_config_directory)
        except FileExistsError:
            shutil.rmtree(self._config.nginx_config_directory)
            logger.warning('Overriding the nginx conf directory '
                            f'"{self._config.nginx_config_directory}"...')
            os.mkdir(self._config.nginx_config_directory)

        # Generate cartesian product for actions space (model configs)
        for model in models:
            model_config_parameters = model.model_config_parameters()

            if model_config_parameters:
                user_model_config_sweeps = \
                    self._run_config_generator.generate_model_config_combinations(
                        model_config_parameters)            

            # user_model_config_sweeps contains all possible model configs for current model 
            # to use as searched action space
            self._run_search_cb = RunSearchCB(self._config, model, user_model_config_sweeps)

            self._execute_vw_search(self._config.iterations)

            self._run_search_cb.save_model()

            self._state_manager.save_checkpoint()

    def admission_control(self, models):
        """
        Runs triton with a wrapping api to determine whether a model will fit
        """
        # Create nginx conf directory if not exists
        try:
            os.mkdir(self._config.nginx_config_directory)
        except FileExistsError:
            shutil.rmtree(self._config.nginx_config_directory)
            logger.warning('Overriding the nginx conf directory '
                            f'"{self._config.nginx_config_directory}"...')
            os.mkdir(self._config.nginx_config_directory)
        
        # Start Triton and Nginx server, and load model variant based on the predicted model_config
        self._nginx.start()
        self._server.start()
        if not self._create_and_load_model_variant(
                original_name=self._run_search_cb.get_model_name(),
                variant_config=model_config):
            self._server.stop()
            self._nginx.stop()

        

        self._server.stop()
        self._nginx.stop()

    def _run_model_no_search(self, model):
        """
        Creates run configs from specified combinations and executes
        them without any run search
        """

        # Generate all the run configs at once and return
        if self._config.triton_launch_mode != 'remote':
            user_model_config_sweeps = \
                self._run_config_generator.generate_model_config_combinations(
                    model.model_config_parameters())
            for user_model_config_sweep in user_model_config_sweeps:
                self._run_config_generator.generate_run_config_for_model_sweep(
                    model, user_model_config_sweep)
        else:
            self._run_config_generator.generate_run_config_for_model_sweep(
                model, None)
        self._execute_run_configs()

    def _run_model_with_search(self, model):
        """
        Searches over the required config elements,
        creates run configs and executes them
        """

        model_config_parameters = model.model_config_parameters()

        # Run config search is enabled, figure out which parameters to sweep over and do sweep
        if self._config.triton_launch_mode == 'remote':
            self._run_model_config_sweep(model, search_model_config=False)
        else:
            if model_config_parameters:
                user_model_config_sweeps = \
                    self._run_config_generator.generate_model_config_combinations(
                        model_config_parameters)
                if model.parameters()['concurrency']:
                    # Both are specified, search over neither
                    for user_model_config_sweep in user_model_config_sweeps:
                        self._run_config_generator.generate_run_config_for_model_sweep(
                            model, user_model_config_sweep)
                    self._execute_run_configs()
                else:
                    # Search through concurrency values only
                    for user_model_config_sweep in user_model_config_sweeps:
                        if self._state_manager.exiting():
                            return
                        self._run_model_config_sweep(
                            model,
                            search_model_config=False,
                            user_model_config_sweep=user_model_config_sweep)
            else:
                # Model Config parameters unspecified
                self._run_model_config_sweep(model, search_model_config=True)

    def _run_model_config_sweep(self,
                                model,
                                search_model_config,
                                user_model_config_sweep=None):
        """
        Initializes the model sweep, iterates until search bounds,
        and executes run configs
        """

        self._run_search.init_model_sweep(model.parameters()['concurrency'],
                                          search_model_config)

        next_model = model
        while not self._state_manager.exiting():

            # Get next model sweep
            next_model, auto_model_config_sweep = self._run_search.get_model_sweep(
                next_model)

            # End search when get_model sweep returns empty
            if not auto_model_config_sweep:
                break
            if user_model_config_sweep:
                model_sweep_for_run_config = user_model_config_sweep
            else:
                model_sweep_for_run_config = auto_model_config_sweep[0]

            self._run_config_generator.generate_run_config_for_model_sweep(
                next_model, model_sweep_for_run_config)
            self._run_search.add_measurements(self._execute_run_configs())

    def _execute_run_configs(self):
        """
        Executes the run configs stored in the run
        config generator until there are none left.
        Returns obtained measurements. Also sends them
        to the result manager
        """

        measurements = []
        while self._run_config_generator.run_configs():
            # Check if exiting
            if self._state_manager.exiting():
                return measurements

            # Remove one run config from the list
            run_config = self._run_config_generator.next_config()

            # If this run config was already run, do not run again, just get the measurement
            measurement = self._get_measurement_if_config_duplicate(run_config)
            if measurement:
                measurements.append(measurement)
                continue

            # Start server, and load model variant
            self._server.start(env=run_config.triton_environment())
            if not self._create_and_load_model_variant(
                    original_name=run_config.model_name(),
                    variant_config=run_config.model_config()):
                self._server.stop()
                continue

            # Profile various batch size and concurrency values.
            # TODO: Need to sort the values for batch size and concurrency
            # for correct measurment of the GPU memory metrics.
            perf_output_writer = None if \
                not self._config.perf_output else FileWriter(self._config.perf_output_path)
            perf_config = run_config.perf_config()

            logger.info(f"Profiling model {perf_config['model-name']}...")
            measurement = self._metrics_manager.profile_model(
                run_config=run_config, perf_output_writer=perf_output_writer)
            if measurement is not None:
                measurements.append(measurement)

            self._server.stop()

        return measurements

    def _execute_vw_search(self, num_iterations):
        """
        Executes the run configs stored in the run
        config generator until there are none left.
        Sends measurements to the result manager
        """

        measurements = []

        if self._config.context_list:
            with open(self._config.context_list, 'r') as context_file:
                dictionary_list = pickle.load(context_file) 
                context_file.close()

        for i in range(1, num_iterations + 1):
            logger.info(f'Iteration: {i}')
            # Generate parameters for current round of evaluation
            if self._config.context_list:
                concurrency = dictionary_list[i - 1]['concurrency-range']
                request_batch_size = dictionary_list[i - 1]['batch-size']
            else:
                concurrency = random.randint(1, self._config.run_config_search_max_concurrency)
                request_batch_size = random.randint(1, self._config.run_config_search_max_preferred_batch_size)

            # Pass context based on passed arguments to vw to get an action
            # Note that keys in context must match those in perf analyzer args
            context = {}
            if 'concurrency-range' in self._config.contexts:
                context['concurrency-range'] = concurrency
            if 'batch-size' in self._config.contexts:
                context['batch-size'] = request_batch_size

            # Check if exiting
            if self._state_manager.exiting():
                self._run_search_cb.save_model()
                return measurements

            # Get predicted model_config and corresponding probability from vw based on the context passed
            model_config, prob = self._run_search_cb.get_vw_predicted_model_config(context)
            logger.info(f'Context: {context}, Action: {model_config.to_dict()}, Prob: {prob}')

            # Get the name of the selected model instance
            current_model_instance_name = model_config.get_field('name')

            analyzer_config = self._config.get_all_config()

            # Generate a perf analyzer config for this run using our context
            perf_config = PerfAnalyzerConfig()
            perf_config_params = {
                'protocol': analyzer_config['client_protocol'],
                'url': 
                    analyzer_config['nginx_http_endpoint']
                    if analyzer_config['client_protocol'] == 'http' else
                    analyzer_config['nginx_grpc_endpoint']
                ,
                'measurement-mode': 'count_windows'
            }
            perf_config.update_config(perf_config_params)

            # This sets the concurrency, batch size, and model name for the the requests
            perf_config.update_config(context)
            perf_config.update_config({'model-name': current_model_instance_name})

            # Update config object for Nginx server
            model_constraints = {}
            model_constraints[current_model_instance_name] = self._run_search_cb.get_model_objectives()
            if 'perf_throughput' in model_constraints[current_model_instance_name].keys():
                # Need to divide perf_throughput by request batch size for nginx rate limiting
                # to account for requests/sec (nginx) vs inferences/sec (triton) difference
                model_constraints[current_model_instance_name]['perf_throughput'] = int(model_constraints[current_model_instance_name]['perf_throughput'] / request_batch_size)
            self._nginx.update_config(model_constraints)

            # Start Triton and Nginx server, and load model variant based on the predicted model_config
            self._nginx.start()
            self._server.start()
            if not self._create_and_load_model_variant(
                    original_name=self._run_search_cb.get_model_name(),
                    variant_config=model_config):
                self._server.stop()
                self._nginx.stop()
                continue

            # Profile various batch size and concurrency values.
            # TODO: Need to sort the values for batch size and concurrency
            # for correct measurement of the GPU memory metrics.
            perf_output_writer = None if \
                not self._config.perf_output else FileWriter()

            # Generate RunConfig from model_config and perf_config
            run_config = RunConfig(self._run_search_cb.get_model_name(), model_config, perf_config, None)

            logger.info(f"Profiling model {current_model_instance_name}...")

            # Only profile model if request batch size is <= max batch size for model
            if int(context['batch-size']) <= int(model_config.to_dict()['maxBatchSize']):
                measurement = self._metrics_manager.profile_model(
                    run_config=run_config, perf_output_writer=perf_output_writer)
            else:
                measurement = None

            # Register cost with CB learner
            self._run_search_cb.register_cost(context, prob, measurement)

            self._server.stop()
            self._nginx.stop()

    def _create_and_load_model_variant(self, original_name, variant_config):
        """
        Creates a directory for the model config
        variant in the output model repository
        """

        variant_name = variant_config.get_field('name')
        if self._config.triton_launch_mode != 'remote':
            model_repository = self._config.model_repository

            original_model_dir = os.path.join(model_repository, original_name)
            new_model_dir = os.path.join(self._output_model_repo_path,
                                         variant_name)
            try:
                # Create the directory for the new model
                os.makedirs(new_model_dir, exist_ok=True)
                variant_config.write_config_to_file(new_model_dir,
                                                    original_model_dir,
                                                    self._last_config_variant)
                self._last_config_variant = os.path.join(
                    self._output_model_repo_path, variant_name)
            except FileExistsError:
                pass

        if self._config.triton_launch_mode != 'c_api':
            self._client.wait_for_server_ready(self._config.client_max_retries)

            if self._client.load_model(model_name=variant_name) == -1:
                return False

            if self._client.wait_for_model_ready(
                    model_name=variant_name,
                    num_retries=self._config.client_max_retries) == -1:
                return False
        return True

    def _get_measurement_if_config_duplicate(self, run_config):
        """
        Checks whether this run config has measurements
        in the state manager's results object
        """

        model_name = run_config.model_name()
        model_config_name = run_config.model_config().get_field('name')
        perf_config_str = run_config.perf_config().representation()

        results = self._state_manager.get_state_variable(
            'ResultManager.results')

        # check whether perf config string is a key in result dict
        if model_name not in results:
            return False
        if model_config_name not in results[model_name]:
            return False
        measurements = results[model_name][model_config_name][1]

        # For backward compatibility with keys that still have -u,
        # we will remove -u from all keys, convert to set and check
        # perf_config_str is present
        if perf_config_str in set(
                map(PerfAnalyzerConfig.remove_url_from_cli_string,
                    measurements.keys())):
            return measurements[perf_config_str]
        else:
            return None
