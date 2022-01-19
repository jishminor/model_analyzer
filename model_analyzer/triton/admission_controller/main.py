from asyncio.windows_events import selector_events
from operator import delitem
from fastapi import FastAPI, File, UploadFile, status, HTTPException
from model_analyzer.entrypoint import get_cli_and_config_options, get_client_handle
import logger
import shutil
import os
import psutil
import csv
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel
from model_analyzer.triton.model.model_config import ModelConfig

args, config = get_cli_and_config_options()
client = get_client_handle(config)

app = FastAPI()

class ModelType(str, Enum):
    tflite = "tflite"
    tf = "tf"

class ConfigGeneration(str, Enum):
    lookup = "lookup"
    lookup_lr = "lookup_lr"
    infer_gcn = "infer_gcn"
    infer_vw = "infer_vw"
    random = "random"

class PerformanceConstraints(BaseModel):
    perf_throughput: float
    perf_latency_p99: float
    # gpu_used_memory: float

class PerformanceObjectives(BaseModel):
    perf_throughput: float
    perf_latency_p99: float
    # gpu_used_memory: float
    # gpu_free_memory: float
    # gpu_utilization: float
    cpu_used_ram: float
    cpu_free_ram: float

class PerformanceTargets(BaseModel):
    constraints: PerformanceConstraints
    objectives: PerformanceObjectives

# Map from class field name to corresponding in model analyzer
# generated csv file
perf_name_map = {
    "perf_throughput": "Throughput (infer/sec)",
    "perf_latency_p99": "p99 Latency (ms)",
    "cpu_used_ram": "RAM Usage (MB)"
}

@app.post("/upload/{model_type}", status_code=status.HTTP_201_CREATED)
async def create_model_file(model_name: str, model_type: ModelType, files: List[UploadFile] = File(...), version: Optional[str] = "1"):
    """
    User uploads a model and it's corresponding triton model configuration.
    Model is created in model repo path specified in model analyzer config.
    """

    if len(files) != 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload 2 files")

    model_file = files[0]
    model_config = files[1]

    new_model_dir = os.path.join(config.model_repository, model_name)
    try:
        # Create the directory for the new model
        os.makedirs(new_model_dir)
    except FileExistsError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model already exists")

    # Write the model file to the newly created directory
    if model_type == ModelType.tflite:
        model_filename = "model.tflite"
    elif model_type == ModelType.tf:
        model_filename = "model.graphdef"
    
    model_file_path = os.path.join(new_model_dir, version, model_filename)
    with open(model_file_path, "wb") as buffer:
        shutil.copyfileobj(model_file.file, buffer)

    # Write the user supplied model configuration to the directory
    model_config_file_path = os.path.join(new_model_dir, "config.pbtxt")
    with open(model_config_file_path, "wb") as buffer:
        shutil.copyfileobj(model_config.file, buffer)

    return {"filename": model_name}

@app.post("/load/{model_name}", status_code=status.HTTP_200_OK)
async def load_model(model_name: str, method: ConfigGeneration, perf_targets: PerformanceTargets, files: Optional[List[UploadFile]] = File(None)):
    try:
        client.wait_for_server_ready(config.client_max_retries)
    except:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Triton server not ready")

    model_dir = os.path.join(config.model_repository, model_name)

    if method == ConfigGeneration.lookup:
        # With the lookup method the user should also attach the model analyzer generated
        # performance output csv file
        model_perf_file_path = os.path.join(model_dir, "perf.csv")
        if files:
            # Write perf csv file if necessary
            if len(files) > 1:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload 2 files")
            perf_csv_file = files[0]
            with open(model_perf_file_path, "wb") as buffer:
                shutil.copyfileobj(perf_csv_file.file, buffer)    

    # Before we load the model we must determine if the model will fit, and if so,
    # the best configuration given the current system state 
    model_config = gen_triton_model_configuration(method, model_dir, perf_targets)

    if not(model_config):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unsatisfiable constraints")

    if client.load_model(model_name=model_name) == -1:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model did not load")

    if client.wait_for_model_ready(
            model_name=model_name,
            num_retries=config.client_max_retries) == -1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model did not become ready")
    
    return f"{model_name} loaded"

@app.post("/unload/{model_name}", status_code=status.HTTP_200_OK)
async def unload_model(model_name: str):
    try:
        client.wait_for_server_ready(config.client_max_retries)
    except:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Triton server not ready")

    if client.unload_model(model_name=model_name) == -1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model did not unload")

    return f"{model_name} unloaded"

def gen_triton_model_configuration(method: ConfigGeneration, model_dir: str, perf_targets: PerformanceTargets):
    """
    Generate the correct directory structure for a triton model and its
    corresponding configuration using one of the four methods listed in
    the ConfigGeneration enum 

    Returns triton model configuration if perf targets satisfiable, 
    else returns None
    """

    # Gather current system state
    cpu_load_5 = psutil.cpu_percent(5)
    free_memory = psutil.virtual_memory()[4]

    model_config = None

    if method == ConfigGeneration.lookup:
        model_config = _get_model_config_lookup(model_dir, perf_targets)
    elif method == ConfigGeneration.lookup_lr:
        model_config = _get_model_config_lookup_lr(model_dir, perf_targets)
    elif method == ConfigGeneration.infer_gcn:
        pass
    elif method == ConfigGeneration.infer_vw:
        pass
    elif method == ConfigGeneration.random:
        pass

    return model_config

def _get_model_config_lookup(model_dir: str, perf_targets: PerformanceTargets):
    """
    Use only the profiled perf data to select a configuration. Closest data point will be used
    to generate the model configuration
    """
    model_perf_file_path = os.path.join(model_dir, "perf.csv")
    with open(model_perf_file_path, newline='') as csvfile:
        perf_reader = csv.DictReader(csvfile, delimiter=',')

        # Iterate through all rows in perf measurements and determine best fit model config
        optimal_row = None
        min_percent_diff = None
        for row in perf_reader:
            # Calculate whether constraints were met by current measurement
            # if not move onto next row
            if (perf_targets.constraints.perf_latency_p99 != None) and (perf_targets.constraints.perf_latency_p99 > row["p99 Latency (ms)"]):
                continue
            if (perf_targets.constraints.perf_throughput != None) and (perf_targets.constraints.perf_throughput < row["Throughput (infer/sec)"]):
                continue

            percent_diff_sum = 0
            # Measure the percent difference between the current measurements
            # and the performance objectives
            for k,v in perf_targets.objectives.dict():
                achieved = row[perf_name_map[k]]
                target = v
                percent_diff_sum += (abs(achieved - target) / ((achieved + target) / 2.0))

            if (min_percent_diff is None) or (percent_diff_sum < min_percent_diff):
                min_percent_diff = percent_diff_sum
                optimal_row = row

        # Generate model config from dict
        model_config_path = os.path.join(model_dir, "config.pbtxt")
        model_config = ModelConfig.create_from_file(model_config_path)

        # Overwrite model config keys with values from selected csv row
        model_config_dict = model_config.get_config()
        for key, value in _get_model_config_params_from_csv(optimal_row):
            if value is not None:
                model_config_dict[key] = value
        model_config = ModelConfig.create_from_dictionary(
            model_config_dict)

def _get_model_config_lookup_lr(model_dir: str, perf_targets: PerformanceTargets):
    """
    Use the profiled perf data to generate a linear regression model for extrapolating the
    best model configuration
    """
    model_perf_file_path = os.path.join(model_dir, "perf.csv")
    with open(model_perf_file_path, newline='') as csvfile:
        perf_reader = csv.DictReader(csvfile, delimiter=',')

def _get_model_config_params_from_csv(row):
    """
    Returns triton model config values from model analyzer csv row
    """

    model_config_params = {}
    model_config_params["instance_group"] = ModelConfig.instance_group_string_to_config(row["Instance Group"])
    model_config_params["dynamic_batching"]["preferred_batch_size"] = ModelConfig.preferred_batch_size_string_to_config(row["Preferred Batch Sizes"])

    return model_config_params