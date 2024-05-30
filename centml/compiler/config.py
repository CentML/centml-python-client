import os
from enum import Enum


class CompilationStatus(Enum):
    NOT_FOUND = "not_found"
    COMPILING = "compiling"
    DONE = "done"


class Config:
    TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    COMPILING_SLEEP_TIME: int = 15

    CACHE_PATH: str = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml"))
    SERVER_IP: str = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
    SERVER_PORT: str = os.getenv("CENTML_SERVER_PORT", default="8090")
    SERVER_URL: str = f"http://{SERVER_IP}:{SERVER_PORT}"

    BACKEND_BASE_PATH: str = os.path.join(CACHE_PATH, "backend")
    SERVER_BASE_PATH: str = os.path.join(CACHE_PATH, "server")

    # Use a constant path since torch.save uses the given file name in it's zipfile.
    # Thus, a different filename would result in a different hash.
    SERIALIZED_MODEL_FILE: str = "serialized_model.zip"
    SERIALIZED_INPUT_FILE: str = "serialized_input.zip"
    PICKLE_PROTOCOL: int = 4

    HASH_CHUNK_SIZE: int = 4096

    # If the server response is smaller than this, don't gzip it
    MINIMUM_GZIP_SIZE: int = 1000


config_instance = Config()
