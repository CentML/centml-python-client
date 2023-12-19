import os
from enum import Enum

class CompilationStatus(Enum):
    NOT_FOUND = 1
    COMPILING = 2
    DONE = 3

class Config:
    TIMEOUT = 10
    MAX_RETRIES = 3
    COMPILING_SLEEP_TIME = 15

    CACHE_PATH = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml"))
    SERVER_IP = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
    SERVER_PORT = os.getenv("CENTML_SERVER_PORT", default="8080")


config_instance = Config()
