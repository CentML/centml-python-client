import os
from enum import Enum
from pydantic_settings import BaseSettings


class CompilationStatus(Enum):
    NOT_FOUND = "NOT_FOUND"
    COMPILING = "COMPILING"
    DONE = "DONE"


class OperationMode(Enum):
    PREDICTION = "PREDICTION"
    REMOTE_COMPILATION = "REMOTE_COMPILATION"


class Config(BaseSettings):
    CENTML_COMPILER_TIMEOUT: int = 10
    CENTML_COMPILER_MAX_RETRIES: int = 3
    CENTML_COMPILER_SLEEP_TIME: int = 15

    CENTML_BASE_CACHE_DIR: str = os.path.expanduser("~/.cache/centml")
    CENTML_BACKEND_BASE_PATH: str = os.path.join(CENTML_BASE_CACHE_DIR, "backend")
    CENTML_SERVER_BASE_PATH: str = os.path.join(CENTML_BASE_CACHE_DIR, "server")

    CENTML_SERVER_URL: str = "http://0.0.0.0:8090"

    # Use a constant path since torch.save uses the given file name in it's zipfile.
    # Using a different filename would result in a different hash.
    CENTML_SERIALIZED_MODEL_FILE: str = "serialized_model.zip"
    CENTML_SERIALIZED_INPUT_FILE: str = "serialized_input.zip"
    CENTML_PICKLE_PROTOCOL: int = 4

    CENTML_HASH_CHUNK_SIZE: int = 4096

    # If the server response is smaller than this, don't gzip it
    CENTML_MINIMUM_GZIP_SIZE: int = 1000

    CENTML_MODE: OperationMode = OperationMode.REMOTE_COMPILATION
    CENTML_PREDICTION_DATA_FILE: str = 'tests/sample_data.csv'
    CENTML_PREDICTION_GPUS: str = "A10G,A100SXM440GB,L4,H10080GBHBM3"
    CENTML_PROMETHEUS_PORT: int = 8000


settings = Config()
