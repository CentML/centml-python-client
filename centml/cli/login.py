import sys
import os
import json
import click

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

    if not cred:
        sys.exit("CentML credentials not found. Please login...")

    return cred['idToken']


@click.command(help="Login to CentML")
@click.argument("token_file", required=False)
def login(token_file):
    if token_file:
        store_centml_cred(token_file)

    if load_centml_cred():
        click.echo(f"Authenticating with credentials from {Config.centml_cred_file}\n")
        click.echo("Login successful")
    else:
        click.echo("Login with CentML authentication token")
        click.echo("Usage: centml login TOKEN_FILE\n")
        choice = click.confirm("Do you want to download the token?")

        if choice:
            click.launch(f"{Config.centml_web_url}?isCliAuthenticated=true")
        else:
            click.echo("Login unsuccessful")


@click.command(help="Logout from CentML")
def logout():
    if os.path.exists(Config.centml_cred_file):
        os.remove(Config.centml_cred_file)
    click.echo("Logout successful")
