import os
from enum import Enum
from urllib.parse import urlparse

class CompilationStatus(Enum):
    NOT_FOUND = "NOT_FOUND"
    COMPILING = "COMPILING"
    DONE = "DONE"


class Config:
    TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    COMPILING_SLEEP_TIME: int = 15

    CACHE_PATH: str = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml"))

    SERVER_URL: str = os.getenv("CENTML_SERVER_URL", default="http://0.0.0.0:8090")
    SERVER_IP: str
    SERVER_PORT: str 
    SERVER_IP, SERVER_PORT = urlparse(SERVER_URL).hostname, urlparse(SERVER_URL).port

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
