import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # It is possible to override the default values by setting the environment variables
    model_config = SettingsConfigDict(env_file=Path('.env'))

    CENTML_WEB_URL: str = os.getenv("CENTML_WEB_URL") or "https://app.centml.com/"
    CENTML_CONFIG_PATH: str = os.getenv("CENTML_CONFIG_PATH") or os.path.expanduser("~/.centml")
    CENTML_CRED_FILE: str = os.getenv("CENTML_CRED_FILE") or "credentials.json"
    CENTML_CRED_FILE_PATH: str = os.path.join(CENTML_CONFIG_PATH,CENTML_CRED_FILE)

    CENTML_PLATFORM_API_URL: str = os.getenv("CENTML_PLATFORM_API_URL") or "https://api.centml.com"

    CENTML_FIREBASE_API_KEY: str = os.getenv("CENTML_FIREBASE_API_KEY") or "AIzaSyChPXy41cIAxS_Nd8oaYKyP_oKkIucobtY" 


settings = Config()