import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # It is possible to override the default values by setting the environment variables
    model_config = SettingsConfigDict(env_file=Path('.env'))

    CENTML_WEB_URL: str = os.getenv("CENTML_WEB_URL", default="https://app.centml.com/")
    CENTML_CONFIG_PATH: str = os.getenv("CENTML_CONFIG_PATH", default=os.path.expanduser("~/.centml"))
    CENTML_CRED_FILE: str = os.getenv("CENTML_CRED_FILE", default="credentials.json")
    CENTML_CRED_FILE_PATH: str = os.path.join(CENTML_CONFIG_PATH, CENTML_CRED_FILE)

    CENTML_PLATFORM_API_URL: str = os.getenv("CENTML_PLATFORM_API_URL", default="https://api.centml.com")

    CENTML_FIREBASE_API_KEY: str = os.getenv(
        "CENTML_FIREBASE_API_KEY", default="AIzaSyChPXy41cIAxS_Nd8oaYKyP_oKkIucobtY"
    )


settings = Config()
