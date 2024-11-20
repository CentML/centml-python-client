import time
import sys
import os
import json
import requests
import jwt

from centml.sdk.config import settings


def refresh_centml_token(refresh_token):
    api_key = settings.CENTML_FIREBASE_API_KEY

    cred = requests.post(
        f"https://securetoken.googleapis.com/v1/token?key={api_key}",
        headers={"content-type": "application/json; charset=UTF-8"},
        data=json.dumps({"grantType": "refresh_token", "refreshToken": refresh_token}),
        timeout=3,
    ).json()

    with open(settings.CENTML_CRED_FILE_PATH, 'w') as f:
        json.dump(cred, f)

    return cred


def store_centml_cred(token_file):
    try:
        with open(token_file, 'r') as f:
            os.makedirs(settings.CENTML_CONFIG_PATH, exist_ok=True)
            refresh_token = json.load(f)["refresh_token"]

            refresh_centml_token(refresh_token)
    except Exception:
        sys.exit(f"Invalid auth token file: {token_file}")


def load_centml_cred():
    cred = None

    if os.path.exists(settings.CENTML_CRED_FILE_PATH):
        with open(settings.CENTML_CRED_FILE_PATH, 'r') as f:
            cred = json.load(f)

    return cred


def get_centml_token():
    cred = load_centml_cred()

    if not cred:
        sys.exit("CentML credentials not found. Please login...")

    exp_time = int(jwt.decode(cred["id_token"], options={"verify_signature": False})["exp"])

    if time.time() >= exp_time - 100:
        cred = refresh_centml_token(cred["refresh_token"])

    return cred["id_token"]


def remove_centml_cred():
    if os.path.exists(settings.CENTML_CRED_FILE_PATH):
        os.remove(settings.CENTML_CRED_FILE_PATH)
