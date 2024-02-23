import os


class Config:
    centml_web_url = "https://main.d1tz9z8hgabab9.amplifyapp.com/"
    centml_config_dir = os.getenv("CENTML_CONFIG_PATH", default=os.path.expanduser("~/.centml"))
    centml_cred_file = centml_config_dir + "/" + os.getenv("CENTML_CRED_FILE", default="credential")

    platformapi_url = "https://api.centml.org"
