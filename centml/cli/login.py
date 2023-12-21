import sys
import os
import webbrowser
import json

from .config import Config


def store_centml_cred(token_file):
    with open(token_file, 'r') as f:
        os.makedirs(Config.centml_config_dir, exist_ok=True)
        data = json.load(f)

        with open(Config.centml_cred_file, 'w') as f:
            json.dump(data, f)


def load_centml_cred():
    cred = None

    if os.path.exists(Config.centml_cred_file):
        with open(Config.centml_cred_file, 'r') as f:
            cred = json.load(f)

    return cred


def get_centml_token():
    cred = load_centml_cred()

    if cred:
        sys.exit("CentML credentials not found!!!")

    return cred['idToken']


def login(token_file):
    if token_file:
        store_centml_cred(token_file)

    if load_centml_cred():
        print(f"Authenticating with credentials from {Config.centml_cred_file}")
        print()
        print('Login successful')
    else:
        print("Login with CentML authentication token")
        print("Usage: centml login <token_file>")
        print()
        print("Do you want to download the token? (Y/n) ", end="")

        cmd_input = input()
        if cmd_input in ('n', 'N'):
            print("Login unsuccessful")
        else:
            webbrowser.open(f"{Config.centml_web_url}?isCliAuthenticated=true")


def logout():
    if os.path.exists(Config.centml_cred_file):
        os.remove(Config.centml_cred_file)
    print("Logout successful")
