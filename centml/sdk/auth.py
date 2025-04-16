import time
import sys
import os
import json
import requests
import jwt

from centml.sdk.config import settings


def refresh_centml_token(refresh_token):
    payload = {
        "client_id": settings.CENTML_WORKOS_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    response = requests.post(
        "https://auth.centml.com/user_management/authenticate",
        headers={"Content-Type": "application/json; charset=UTF-8"},
        json=payload,
        timeout=3,
    )
    response_dict = response.json()

    # If there is an error, we should remove the credentials and the user needs to sign in again.
    if "error" in response_dict:
        if os.path.exists(settings.CENTML_CRED_FILE_PATH):
            os.remove(settings.CENTML_CRED_FILE_PATH)
        cred = None
    else:
        cred = {key: response_dict[key] for key in ("access_token", "refresh_token") if key in response_dict}
        with open(settings.CENTML_CRED_FILE_PATH, "w") as f:
            json.dump(cred, f)

    return cred


def store_centml_cred(token_file):
    try:
        with open(token_file, "r") as f:
            os.makedirs(settings.CENTML_CONFIG_PATH, exist_ok=True)
            refresh_token = json.load(f)["refresh_token"]

            refresh_centml_token(refresh_token)
    except Exception:
        sys.exit(f"Invalid auth token file: {token_file}")


def load_centml_cred():
    cred = None

    if os.path.exists(settings.CENTML_CRED_FILE_PATH):
        with open(settings.CENTML_CRED_FILE_PATH, "r") as f:
            cred = json.load(f)

    return cred


def get_centml_token():
    cred = load_centml_cred()
    if not cred:
        sys.exit("CentML credentials not found. Please login...")
    exp_time = int(jwt.decode(cred["access_token"], options={"verify_signature": False})["exp"])

    if time.time() >= exp_time - 100:
        cred = refresh_centml_token(cred["refresh_token"])
        if cred is None:
            sys.exit("Could not refresh credentials. Please login and try again...")

    return cred["access_token"]


def remove_centml_cred():
    if os.path.exists(settings.CENTML_CRED_FILE_PATH):
        os.remove(settings.CENTML_CRED_FILE_PATH)
