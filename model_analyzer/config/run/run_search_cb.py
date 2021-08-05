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
import logging

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
        self._encode_context_numeric = config.encode_context_numeric
        self._encode_actions_numeric = config.encode_actions_numeric
        self._model_repository = config.get_all_config()['model_repository']

        # Filter out invalid model config sweeps from passed list
        self._remove_known_invalid_model_configs()

        # Holds most recently selected active model config index
        self._active_model_config_index = 0

        # Shuffle the model config sweeps
        random.shuffle(self._user_model_config_sweeps)

        # Instantiate learner in VW for online learning
        vw_string = ''
        if self._adf:
            vw_string += '--cb_explore_adf -q CA '
        else:
            vw_string = '--cb_explore {len(self._user_model_config_sweeps)} ')

        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            vw_string += '--quiet '
        
        if config.exploration == 'epsilon':
            vw_string += f'--epsilon {config.epsilon}'
        elif config.exploration == 'first':
            vw_string += f'--first {config.tau}'
        elif config.exploration == 'bag':
            vw_string += f'--bag {config.policies}'
        elif config.exploration == 'cover':
            vw_string += f'--cover {config.policies}'
        elif config.exploration == 'softmax':
            vw_string += f'--softmax --lambda {config.lambda_value}'
        elif config.exploration == 'rnd':
            vw_string += f'--rnd {config.rnd} --epsilon {config.epsilon}'

        self._vw = pyvw.vw(vw_string) 

    def _remove_known_invalid_model_configs(self):
        # Remove all items in actions list where max batch size < preferred_batchsize
        self._user_model_config_sweeps = [x for x in self._user_model_config_sweeps if int(x['max_batch_size']) >= int(x['dynamic_batching']['preferred_batch_size'][0])]

        # Remove all items where both cpu and gpu instances are 0
        # self._user_model_config_sweeps = [x for x in self._user_model_config_sweeps if x[1] > 0 or x[4] > 0]

        # Remove all items where cpu > 0 and tensorrt is True
        # self._user_model_config_sweeps = [x for x in self._user_model_config_sweeps if not(x[1] > 0 and x[5])]

        # Account for triton bug where tensorrt can't be applied to model running on both cpu and gpu
        # self._user_model_config_sweeps = [x for x in self._user_model_config_sweeps if not(
            # x[1] > 0 and x[4] > 0 and x[5])]

    def get_model_name(self):
        return self._model.model_name()

    def get_vw_predicted_model_config(self, context):
        """
        Fetch the run config from the VW predicted best model_config
        """

        vw_text = self._to_vw_example_format(context)
        pmf = self._vw.predict(vw_text)
        logging.debug(pmf)
        self._active_model_config_index, prob = self._sample_custom_pmf(pmf)
        model_sweep = self._user_model_config_sweeps[self._active_model_config_index]
        model_config = self._generate_model_config_from_sweep(model_sweep)

        return model_config, prob

    def get_random_model_config(self):
        """
        Get a random run config
        """

        self._active_model_config_index = random.randint(0, len(self._user_model_config_sweeps) - 1)
        model_sweep = self._user_model_config_sweeps[self._active_model_config_index]
        model_config = self._generate_model_config_from_sweep(model_sweep)
        return model_config, 1 / len(self._user_model_config_sweeps)

    def _generate_model_config_from_sweep(self, model_sweep):
        # Generate ModelConfig from model_config_param dict (model_sweep)
        model_config = ModelConfig.create_from_file(
            f'{self._model_repository}/{self._model.model_name()}')
        model_tmp_name = f'{self._model.model_name()}_i{self._active_model_config_index}'

        # Overwrite model config keys with values from model_sweep
        model_config_dict = model_config.get_config()
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

        if measurement:
            costs = {}
            for key, value in self._model.objectives().items():
                logging.info(f'Target: {key}, Desired: {value}, Achieved: {measurement.get_metric(key).value()}')
                costs[key] = abs(measurement.get_metric(key).value() - value)

            # Sum all costs to generate final cost
            cost = sum(costs.values())
        else:
            # If measurement came back None, give max penalty
            cost = 1000

        logging.info(f'Cost is {cost}')

        if not(self._no_learn):
            # Inform VW of what happened so we can learn from it
            vw_format = self._vw.parse(self._to_vw_example_format(context, (cost, prob)), pyvw.vw.lContextualBandit)
            # Learn
            self._vw.learn(vw_format)
            # Let VW know you're done with these objects
            self._vw.finish_example(vw_format)
    
    def _to_vw_example_format(self, context, cb_label=None):
        """
        This function modifies (context, model_config, cost, probability) to VW friendly format
        """

        if cb_label is not None:
            cost, prob = cb_label

        chosen_model_sweep = self._user_model_config_sweeps[self._active_model_config_index]

        # Generate context string
        context_string = ""
        for k, v in context.items():
            context_string += k.replace('-', '_')
            
            if (type(v) == int or type(v) == float) and self._encode_context_numeric:
                context_string += f':{v} '
            else:
                # Here we always use '=' to encode as a string, as opposed to ':' which
                # encodes numeric values. It seems avg loss is better this way
                context_string += f'={v} '

        action_string = ""
        if self._adf:
            # Generate adf formatted cb data
            action_string += f'shared |Context {context_string}\n'
            for model_sweep in self._user_model_config_sweeps:
                if cb_label is not None and model_sweep == chosen_model_sweep:
                    action_string += f'0:{cost}:{prob} '

                delimiter = ':' if self._encode_actions_numeric else '='
                action_string += f'|Action '
                for k, v in model_sweep.items():
                    if k == 'max_batch_size':
                        action_string += k
                        action_string += f'{delimiter}{v} '
                    if k == 'dynamic_batching':
                        for k, v in v.items():
                            if k == 'preferred_batch_size':
                                action_string += k
                                action_string += f'{delimiter}{v[0]} '
                            if k == 'max_queue_delay_microseconds':
                                action_string += k
                                action_string += f'{delimiter}{v} '
                    if k == 'instance_group':
                        for instance_group in v:
                            if instance_group['kind'] == 'KIND_CPU':
                                action_string += 'cpu_instances'
                                action_string += f'{delimiter}{instance_group["count"]} '
                            if instance_group['kind'] == 'KIND_GPU':
                                action_string += 'gpu_instances'
                                action_string += f'{delimiter}{instance_group["count"]} '
                action_string += '\n'

        else:
            # Generate standard formatted cb data
            if cb_label is not None:
                # VW doesn't allow model_config id of 0
                action_string += f'{self._active_model_config_index + 1}:{cost}:{prob} '
            action_string += '| ' + f'{context_string}\n'

        # Strip the last newline
        logging.debug(action_string)
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
    