import os

class Config:
    centml_web_url = "https://pr-178.d193t9o092mgh6.amplifyapp.com/"
    centml_config_dir = os.getenv("CENTML_CONFIG_PATH", default=os.path.expanduser("~/.centml"))
    centml_cred_file = centml_config_dir + "/" + os.getenv("CENTML_CRED_FILE", default="credential")
