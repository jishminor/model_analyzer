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
from model_analyzer.constants import THROUGHPUT_GAIN
import random

from vowpalwabbit import pyvw


class RunSearchCB():
    """
    A class responsible for searching the config space using CB techniques.
    """

    def __init__(self, config, model_configs):
        self._max_concurrency = config.run_config_search_max_concurrency
        self._max_instance_count = config.run_config_search_max_instance_count
        self._max_preferred_batch_size = config.run_config_search_max_preferred_batch_size
        self._model_configs = model_configs
        self._no_learn = config.no_learn
        self._adf = config.adf
        self._epsilon = config.epsilon

        # Instantiate learner in VW for online learning
        if self._adf:
            self._vw = pyvw.vw("--cb_explore_adf -q CA --epsilon {}".format(self._epsilon))
        else:
            self._vw = pyvw.vw(
                "--cb_explore {} --epsilon {}".format(len(self._model_configs), self._epsilon))

    def get_vw_predicted_model_config(self, model, context):
        """
        Fetch the run config from the VW predicted best model_config
        """

        vw_text = model.to_vw_example_format(context)
        pmf = self._vw.predict(vw_text)
        chosen_model_config_index, prob = self._sample_custom_pmf(pmf)
        return self._model_configs[chosen_model_config_index], prob

    def get_random_model_config(self):
        """
        Get a random run config
        """

        chosen_model_config_index = random.randint(0, len(self._model_configs) + 1)
        return self._model_configs[chosen_model_config_index], 1 / len(self._model_configs)

    def register_cost(self, model, model_config, context, prob, measurement):
        """
        Register cost of measurement generated from profiling model with VW
        Cost is measured as delta from objectives
        """

        costs = {}
        for key, value in model.objectives().items():
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
            chosen_model_config, cost, prob = cb_label

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
            for model_config in self._model_configs:
                if cb_label is not None and model_config == chosen_model_config:
                    action_string += "0:{}:{} ".format(cost, prob)

                action_string += "|Action "
                for k, v in model_config.items():
                    action_string += k
                    if type(v) == int or type(v) == float:
                        action_string += ":{} ".format(v)
                    else:
                        action_string += "={} ".format(v)

        else:
            # Generate standard formatted cb data
            if cb_label is not None:
                chosen_model_config_index = self._model_configs.index(chosen_model_config)
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
    