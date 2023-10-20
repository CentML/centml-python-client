import os


class Config:
    TIMEOUT = 10
    TIMEOUT_COMPILE = 100
    MAX_RETRIES = 3

    SERVER_IP = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
    SERVER_PORT = os.getenv("CENTML_SERVER_PORT", default="8080")
