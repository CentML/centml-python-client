import os
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    CENTML_WEB_URL: str = "https://app.centml.com/"
    CENTML_CONFIG_PATH: str = os.path.expanduser("~/.centml")
    CENTML_CRED_FILE: str = "credential"
    CENTML_CRED_FILE_PATH: str = CENTML_CONFIG_PATH + "/" + CENTML_CRED_FILE

    CENTML_PLATFORM_API_URL: str = "https://api.centml.com"

    CENTML_FIREBASE_API_KEY: str = "AIzaSyChPXy41cIAxS_Nd8oaYKyP_oKkIucobtY"


settings = Config()
