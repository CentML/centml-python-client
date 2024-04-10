import os
import shutil
from centml.compiler.config import config_instance


def get_backend_compiled_forward_path(model_id: str):
    os.makedirs(config_instance.BACKEND_BASE_PATH, exist_ok=True)
    return os.path.join(config_instance.BACKEND_BASE_PATH, model_id, "compilation_return.pkl")


def get_server_compiled_forward_path(model_id: str):
    os.makedirs(config_instance.SERVER_BASE_PATH, exist_ok=True)
    return os.path.join(config_instance.SERVER_BASE_PATH, model_id, "compilation_return.pkl")


# This function will delete the storage_path/{model_id} directory
def dir_cleanup(model_id: str):
    dir_path = os.path.join(config_instance.SERVER_BASE_PATH, model_id)
    if not os.path.exists(dir_path):
        return  # Directory does not exist, return

    if not os.path.isdir(dir_path):
        raise Exception(f"'{dir_path}' is not a directory")

    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        raise Exception("Failed to delete the directory") from e
