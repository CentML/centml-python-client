import time
import sys
import os
import json
import requests
import jwt

from .config import Config


def refresh_centml_token(refresh_token):
    api_key = Config.firebase_api_key

    cred = requests.post(
        f"https://securetoken.googleapis.com/v1/token?key={api_key}",
        headers={"content-type": "application/json; charset=UTF-8"},
        data=json.dumps({"grantType": "refresh_token", "refreshToken": refresh_token}),
        timeout=3,
    ).json()

    with open(Config.centml_cred_file, 'w') as f:
        json.dump(cred, f)

    return cred


def store_centml_cred(token_file):
    with open(token_file, 'r') as f:
        os.makedirs(Config.centml_config_dir, exist_ok=True)
        refresh_token = json.load(f)["refreshToken"]

        refresh_centml_token(refresh_token)


def load_centml_cred():
    cred = None

    if os.path.exists(Config.centml_cred_file):
        with open(Config.centml_cred_file, 'r') as f:
            cred = json.load(f)

    return cred


def get_centml_token():
    cred = load_centml_cred()

    if not cred:
        sys.exit("CentML credentials not found. Please login...")

    exp_time = int(jwt.decode(cred["access_token"], options={"verify_signature": False})["exp"])

    if time.time() >= exp_time - 100:
        cred = refresh_centml_token(cred["refresh_token"])

    return cred["access_token"]


def remove_centml_cred():
    if os.path.exists(Config.centml_cred_file):
        os.remove(Config.centml_cred_file)
