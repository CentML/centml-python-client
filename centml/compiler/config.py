import os
from enum import Enum
from pydantic_settings import BaseSettings


class CompilationStatus(Enum):
    NOT_FOUND = "NOT_FOUND"
    COMPILING = "COMPILING"
    DONE = "DONE"


class Config(BaseSettings):
    TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    COMPILING_SLEEP_TIME: int = 15

    CENTML_CACHE_DIR: str = os.path.expanduser("~/.cache/centml")
    BACKEND_BASE_PATH: str = os.path.join(CENTML_CACHE_DIR, "backend")
    SERVER_BASE_PATH: str = os.path.join(CENTML_CACHE_DIR, "server")

    CENTML_SERVER_URL: str = "http://0.0.0.0:8090"

    # Use a constant path since torch.save uses the given file name in it's zipfile.
    # Using a different filename would result in a different hash.
    SERIALIZED_MODEL_FILE: str = "serialized_model.zip"
    SERIALIZED_INPUT_FILE: str = "serialized_input.zip"
    PICKLE_PROTOCOL: int = 4

    HASH_CHUNK_SIZE: int = 4096

    # If the server response is smaller than this, don't gzip it
    MINIMUM_GZIP_SIZE: int = 1000


settings = Config()
