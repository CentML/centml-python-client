import os
import shutil
from centml.compiler.config import settings


def get_backend_compiled_forward_path(model_id: str):
    os.makedirs(os.path.join(settings.BACKEND_BASE_PATH, model_id), exist_ok=True)
    return os.path.join(settings.BACKEND_BASE_PATH, model_id, "compilation_return.pkl")


def get_server_compiled_forward_path(model_id: str):
    os.makedirs(os.path.join(settings.SERVER_BASE_PATH, model_id), exist_ok=True)
    return os.path.join(settings.SERVER_BASE_PATH, model_id, "compilation_return.pkl")


# This function will delete the storage_path/{model_id} directory
def dir_cleanup(model_id: str):
    dir_path = os.path.join(settings.SERVER_BASE_PATH, model_id)
    if not os.path.exists(dir_path):
        return  # Directory does not exist, return

    if not os.path.isdir(dir_path):
        raise Exception(f"'{dir_path}' is not a directory")

    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        raise Exception("Failed to delete the directory") from e


def verify_model_and_input_paths(model_path, input_path):
    if not model_path or not input_path:
        raise Exception("Model or inputs not serialized")
    if not os.path.isfile(model_path):
        raise Exception(f"Model not saved at path {model_path}")
    if not os.path.isfile(input_path):
        raise Exception(f"Inputs not saved at path {input_path}")
