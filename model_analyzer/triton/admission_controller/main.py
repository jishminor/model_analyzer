from fastapi import FastAPI, File, UploadFile, status, HTTPException
from model_analyzer.entrypoint import get_cli_and_config_options, get_client_handle
import logger
import shutil
import os
import psutil
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

args, config = get_cli_and_config_options()
client = get_client_handle(config)

app = FastAPI()

class ModelType(str, Enum):
    tflite = "tflite"
    tf = "tf"

class ConfigGeneration(str, Enum):
    lookup = "lookup"
    infer_gcn = "infer_gcn"
    infer_vw = "infer_vw"
    random = "random"

class PerformanceConstraints(BaseModel):
    perf_throughput: int
    perf_latency_p99: int
    gpu_used_memory: int

class PerformanceObjectives(BaseModel):
    perf_throughput: int
    perf_latency_p99: int
    gpu_used_memory: int
    gpu_free_memory: int
    gpu_utilization: int
    cpu_used_ram: int
    cpu_free_ram: int

class PerformanceTargets(BaseModel):
    constraints: PerformanceConstraints
    objectives: PerformanceObjectives

@app.get("/")
async def root():
    return {"message": "Hello World"}

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
async def load_model(model_name: str, method: ConfigGeneration, perf_targets: PerformanceTargets):
    try:
        client.wait_for_server_ready(config.client_max_retries)
    except:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Triton server not ready")

    # Before we load the model we must determine if the model will fit, and if so,
    # the best configuration given the current system state 
    gen_triton_model_configuration(method, model_name, perf_targets)   

    if client.load_model(model_name=model_name) == -1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model did not load")

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

def gen_triton_model_configuration(method: ConfigGeneration, model_name: str, perf_targets: PerformanceTargets):
    """
    Generate the correct directory structure for a triton model and its
    corresponding configuration using one of the four methods listed in
    the ConfigGeneration enum 

    Returns triton model configuration if perf targets satisfiable, 
    else returns None
    """

    cpu_load_5 = psutil.cpu_percent(5)

    free_memory = psutil.virtual_memory()[4]

    model_config_file_path = os.path.join(config.model_repository, model_name, "config.pbtxt")



    if method == ConfigGeneration.lookup:
        pass
    elif method == ConfigGeneration.infer_gcn:
        pass
    elif method == ConfigGeneration.infer_vw:
        pass
    elif method == ConfigGeneration.random:
        pass

    return None
