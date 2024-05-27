import os
from enum import Enum
import hidet

hidet.option.imperative(False)


class CompilationStatus(Enum):
    NOT_FOUND = "not_found"
    COMPILING = "compiling"
    DONE = "done"


class Config:
    TIMEOUT = 10
    MAX_RETRIES = 3
    COMPILING_SLEEP_TIME = 15

    CACHE_PATH = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml"))
    SERVER_IP = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
    SERVER_PORT = os.getenv("CENTML_SERVER_PORT", default="8080")
    SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

    BACKEND_BASE_PATH = os.path.join(CACHE_PATH, "backend")
    SERVER_BASE_PATH = os.path.join(CACHE_PATH, "server")

    # Use a constant path since torch.save uses the given file name in it's zipfile.
    # Thus, a different path would result in a different hash.
    SERIALIZED_MODEL_FILE = "serialized_model.pkl"
    SERIALIZED_INPUT_FILE = "serialized_input.pkl"
    PICKLE_PROTOCOL = 4


config_instance = Config()
