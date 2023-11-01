import os


class Config:
    TIMEOUT = 10
    TIMEOUT_COMPILE = 1000
    MAX_RETRIES = 0
    COMPILING_SLEEP_TIME = 15

    CACHE_PATH = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml"))
    SERVER_IP = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
    SERVER_PORT = os.getenv("CENTML_SERVER_PORT", default="8080")
