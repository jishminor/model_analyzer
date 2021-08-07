# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

from model_analyzer.model_analyzer_exceptions \
    import TritonModelAnalyzerException

from jinja2 import Template

class NginxServerConfig:
    """
    A config class to set arguments to the Nginx
    Server in front of Triton Server. An argument set to None will use the server default.
    """

    nginx_conf_template = '''
    
    worker_processes auto;
    pid /run/nginx.pid;
    events {
            worker_connections 768;
    }
    http {
            #### BEGIN: Stuff from the default nginx.conf ####
            tcp_nopush on;
            tcp_nodelay on;
            keepalive_timeout 70;  # Bumped from 65
            types_hash_max_size 2048;
            include /etc/nginx/mime.types;
            default_type application/octet-stream;
            ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
            ssl_prefer_server_ciphers on;
            access_log /var/log/nginx/access.log;
            error_log /var/log/nginx/error.log;
            #### END Stuff from the default nginx.conf ####
            gzip on;
            ################
            # TritonServer #
            ################
            
            # Enable rate limiting
            {% for k,v in model_constraints.items() %}
            limit_req_zone $binary_remote_addr zone={{ k }}_limit:4m rate={{ v['perf_throughput'] }}r/s;
            {% endfor %}

            server {
                    listen {{ analyzer_config['nginx_http_endpoint'].split(':')[-1] }} default_server;
                    listen [::]:{{ analyzer_config['nginx_http_endpoint'].split(':')[-1] }} default_server;
                    server_name tritonserver;
                    
                    # Reverse-proxy to tritonserver
                    location / {
                            proxy_pass        http://{{ analyzer_config['triton_http_endpoint'] }};
                    }

                    {% for k,v in model_constraints.items() %}
                    location /v2/models/{{ k }}/infer {
                            proxy_pass        http://{{ analyzer_config['triton_http_endpoint'] }}/v2/models/{{ k }}/infer;
                            limit_req         zone={{ k }}_limit burst=20;
                            client_max_body_size 100M; # This needs to be set dynamically based on request size
                    }
                    {% endfor %}
            }

            server {
                    listen {{ analyzer_config['nginx_grpc_endpoint'].split(':')[-1] }} http2 default_server;
                    listen [::]:{{ analyzer_config['nginx_grpc_endpoint'].split(':')[-1] }} http2 default_server;
                    server_name tritonservergrpc;
                    proxy_buffering off;

                    # Reverse-proxy to tritonserver
                    location /inference.GRPCInferenceService {
                            grpc_pass         grpc://inference_service;
                    }

                    location /inference.GRPCInferenceService/ModelInfer {
                            grpc_pass         grpc://inference_service;
                            {% for k,v in model_constraints.items() %}
                            limit_req         zone={{ k }}_limit burst=20;
                            {% endfor %}
                            client_max_body_size 100M;
                    }
            }

            # Backend gRPC servers
            #
            upstream inference_service {
                    zone inference_service 64k;
                    server {{ analyzer_config['triton_grpc_endpoint'] }};
            }
    }
    '''

    def __init__(self, model_constraints, analyzer_config, config_path='/etc/nginx/triton-nginx.conf'):
        """
        Construct NginxServerConfig
        """

        # TODO: pass input tensor byte size in the constraints for use
        self._model_constraints = model_constraints
        self._analyzer_config = analyzer_config
        self._config_path = config_path

    def to_nginx_config(self):
        """
        Utility function to convert a config into an nginx config

        Returns
        -------
        str
            the file contents of the nginx config.
        """

        template = Template(self.nginx_conf_template, trim_blocks=True,
                            lstrip_blocks=True, keep_trailing_newline=True)
        return template.render(model_constraints=self._model_constraints, analyzer_config=self._analyzer_config)

    def get_config_path(self):
        return self._config_path

    def get_analyzer_config(self):
        return self._analyzer_config
