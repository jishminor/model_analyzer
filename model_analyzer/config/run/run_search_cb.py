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

from model_analyzer.config.input.objects.config_model_profile_spec \
    import ConfigModelProfileSpec
from model_analyzer.triton.model.model_config import ModelConfig
import random

from vowpalwabbit import pyvw


class RunSearchCB():
    """
    A class responsible for searching the config space using CB techniques.
    """

    def __init__(self, config, model, user_model_config_sweeps):
        self._user_model_config_sweeps = user_model_config_sweeps
        self._model = model
        self._no_learn = config.no_learn
        self._adf = config.adf
        self._epsilon = config.epsilon
        self._model_repository = config.get_all_config()['model_repository']

        # Instantiate learner in VW for online learning
        if self._adf:
            self._vw = pyvw.vw("--cb_explore_adf -q CA --epsilon {}".format(self._epsilon))
        else:
            self._vw = pyvw.vw(
                "--cb_explore {} --epsilon {}".format(len(self._user_model_config_sweeps), self._epsilon))
        self._base_model_config = ModelConfig.create_from_file(
            f'{self._model_repository}/{self._model.model_name()}')

    def get_model_name(self):
        return self._model.model_name()

    def get_vw_predicted_model_config(self, context):
        """
        Fetch the run config from the VW predicted best model_config
        """

        vw_text = self._to_vw_example_format(context)
        pmf = self._vw.predict(vw_text)
        chosen_model_config_index, prob = self._sample_custom_pmf(pmf)

        model_sweep = self._user_model_config_sweeps[chosen_model_config_index]
        model_config = self._generate_model_config_from_sweep(model_sweep, chosen_model_config_index)

        return model_config, prob

    def get_random_model_config(self):
        """
        Get a random run config
        """

        chosen_model_config_index = random.randint(0, len(self._user_model_config_sweeps) + 1)
        model_sweep = self._user_model_config_sweeps[chosen_model_config_index]
        model_config = self._generate_model_config_from_sweep(model_sweep, chosen_model_config_index)
        return model_config, 1 / len(self._user_model_config_sweeps)

    def _generate_model_config_from_sweep(self, model_sweep, index):
        # Generate ModelConfig from model_config_param dict (model_sweep)
        model_tmp_name = f'{self._model.model_name()}_i{index}'

        # Overwrite model config keys with values from model_sweep
        model_config_dict = self._base_model_config.get_config()
        for key, value in model_sweep.items():
            if value is not None:
                model_config_dict[key] = value
        model_config = ModelConfig.create_from_dictionary(
            model_config_dict)
        
        model_config.set_field('name', model_tmp_name)
        model_config.set_cpu_only(self._model.cpu_only())

        return model_config

    def register_cost(self, model_config, context, prob, measurement):
        """
        Register cost of measurement generated from profiling model with VW
        Cost is measured as delta from objectives
        """

        costs = {}
        for key, value in self._model.objectives().items():
            costs[key] = abs(measurement.get_metric(key).value() - value)

        # Sum all costs to generate final cost
        cost = sum(costs.values())

        if not(self._no_learn):
            # Inform VW of what happened so we can learn from it
            vw_format = self._vw.parse(self._to_vw_example_format(context, (model_config, cost, prob)), pyvw.vw.lContextualBandit)
            # Learn
            self._vw.learn(vw_format)
            # Let VW know you're done with these objects
            self._vw.finish_example(vw_format)
    
    def _to_vw_example_format(self, context, cb_label=None):
        """
        This function modifies (context, model_config, cost, probability) to VW friendly format
        """

        if cb_label is not None:
            chosen_model_sweep, cost, prob = cb_label

        # Generate context string
        context_string = ""
        for k, v in context.items():
            context_string += k
            if type(v) == int or type(v) == float:
                context_string += ":{} ".format(v)
            else:
                context_string += "={} ".format(v)

        action_string = ""
        if self._adf:
            # Generate adf formatted cb data
            action_string += "shared |Context {}\n".format(context_string)
            for model_sweep in self._user_model_config_sweeps:
                if cb_label is not None and model_sweep == chosen_model_sweep:
                    action_string += "0:{}:{} ".format(cost, prob)

                action_string += "|Action "
                for k, v in model_sweep.items():
                    if k == 'max_batch_size':
                        action_string += k
                        action_string += ":{} ".format(v)
                    if k == 'dynamic_batching':
                        for k, v in v.items():
                            if k == 'preferred_batch_size':
                                action_string += k
                                action_string += ":{} ".format(v)
                            if k == 'max_queue_delay_microseconds':
                                action_string += k
                                action_string += ":{} ".format(v)
                    if k == 'instance_group':
                        for instance_group in v:
                            if instance_group['kind'] == 'KIND_CPU':
                                action_string += 'cpu_instances'
                                action_string += ":{} ".format(instance_group['count'])
                            if instance_group['kind'] == 'KIND_GPU':
                                action_string += 'gpu_instances'
                                action_string += ":{} ".format(instance_group['count'])

        else:
            # Generate standard formatted cb data
            if cb_label is not None:
                chosen_model_config_index = self._user_model_config_sweeps.index(chosen_model_config)
                # VW doesn't allow model_config id of 0
                action_string += "{}:{}:{} ".format(
                    chosen_model_config_index + 1, cost, prob)
            action_string += "| " + "{}\n".format(context_string)
        # Strip the last newline
        return action_string[:-1]

    def _sample_custom_pmf(self, pmf):
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if(sum_prob > draw):
                return index, prob
    