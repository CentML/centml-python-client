import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # It is possible to override the default values by setting the environment variables
    model_config = SettingsConfigDict(env_file=Path(".env"))

    CENTML_WEB_URL: str = os.getenv("CENTML_WEB_URL", default="https://app.centml.com/")
    CENTML_CONFIG_PATH: str = os.getenv("CENTML_CONFIG_PATH", default=os.path.expanduser("~/.centml"))
    CENTML_CRED_FILE: str = os.getenv("CENTML_CRED_FILE", default="credentials.json")
    CENTML_CRED_FILE_PATH: str = os.path.join(CENTML_CONFIG_PATH, CENTML_CRED_FILE)

    CENTML_PLATFORM_API_URL: str = os.getenv("CENTML_PLATFORM_API_URL", default="https://api.centml.com")

    CENTML_WORKOS_CLIENT_ID: str = os.getenv("CENTML_WORKOS_CLIENT_ID", default="client_01JP5TWW2997MF8AYQXHJEGYR0")


settings = Config()
