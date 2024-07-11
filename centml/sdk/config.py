import os
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    centml_web_url: str = "https://main.d1tz9z8hgabab9.amplifyapp.com/"
    centml_config_dir: str = os.getenv("CENTML_CONFIG_PATH", default=os.path.expanduser("~/.centml"))
    centml_cred_file: str = centml_config_dir + "/" + os.getenv("CENTML_CRED_FILE", default="credential")

    platformapi_url: str = "https://api.centml.org"

    firebase_api_key: str = "AIzaSyBXSNjruNdtypqUt_CPhB8QNl8Djfh5RXI"

settings = Config()