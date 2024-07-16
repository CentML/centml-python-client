import os
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    CENTML_WEB_URL: str = "https://main.d1tz9z8hgabab9.amplifyapp.com/"
    CENTML_CONFIG_PATH: str = os.path.expanduser("~/.centml")
    CENTML_CRED_FILE: str = "credential"
    CENTML_CRED_FILE_PATH: str = CENTML_CONFIG_PATH + "/" + CENTML_CRED_FILE

    PLATFORM_API_URL: str = "https://api.centml.org"

    FIREBASE_API_KEY: str = "AIzaSyBXSNjruNdtypqUt_CPhB8QNl8Djfh5RXI"


settings = Config()
