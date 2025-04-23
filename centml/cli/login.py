import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import secrets
import urllib.parse
import webbrowser

import click
import requests


from centml.sdk import auth
from centml.sdk.config import settings


CLIENT_ID = settings.CENTML_WORKOS_CLIENT_ID
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 57983
REDIRECT_URI = f"http://{SERVER_HOST}:{SERVER_PORT}/callback"
AUTHORIZE_URL = "https://auth.centml.com/user_management/authorize"
AUTHENTICATE_URL = "https://auth.centml.com/user_management/authenticate"
PROVIDER = "authkit"


def generate_pkce_pair():
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def build_auth_url(client_id, redirect_uri, challenge):
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "provider": PROVIDER,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        self.server.auth_code = params.get("code", [None])[0]

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            """
            <html>
                <body>
                    <h1>Succesfully logged into CentML CLI</h1>
                    <p>You can now close this tab and continue in the CLI.</p>
                </body>
            </html>
            """.encode(
                "utf-8"
            )
        )

    def log_message(self, format, *args):
        # Override this to suppress logging
        pass


def get_auth_code():
    server = HTTPServer((SERVER_HOST, SERVER_PORT), OAuthHandler)
    server.handle_request()
    return server.auth_code


def exchange_code_for_token(code, code_verifier):
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": code_verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(AUTHENTICATE_URL, data=data, headers=headers, timeout=3)
    response.raise_for_status()
    return response.json()


@click.command(help="Login to CentML")
@click.argument("token_file", required=False)
def login(token_file):
    if token_file:
        auth.store_centml_cred(token_file)

    cred = auth.load_centml_cred()
    if cred is not None and auth.refresh_centml_token(cred.get("refresh_token")):
        click.echo("Authenticating with stored credentials...\n")
        click.echo("✅ Login successful")
    else:
        click.echo("Logging into CentML...")

        choice = click.confirm("Do you want to log in with your browser now?", default=True)
        if choice:
            try:
                # PKCE Flow
                code_verifier, code_challenge = generate_pkce_pair()
                auth_url = build_auth_url(CLIENT_ID, REDIRECT_URI, code_challenge)
                click.echo("A browser window will open for you to authenticate.")
                click.echo("If it doesn't open automatically, you can copy and paste this URL:")
                click.echo(f"   {auth_url}\n")
                webbrowser.open(auth_url)
                click.echo("Waiting for authentication...")

                code = get_auth_code()
                response_dict = exchange_code_for_token(code, code_verifier)
                # If there is an error, we should remove the credentials and the user needs to sign in again.
                if "error" in response_dict:
                    click.echo("Login failed. Please try again.")
                else:
                    cred = {
                        key: response_dict[key] for key in ("access_token", "refresh_token") if key in response_dict
                    }
                    os.makedirs(os.path.dirname(settings.CENTML_CRED_FILE_PATH), exist_ok=True)
                    with open(settings.CENTML_CRED_FILE_PATH, "w") as f:
                        json.dump(cred, f)
                    click.echo("✅ Login successful")
            except Exception as e:
                click.echo(f"Login failed: {e}")
        else:
            click.echo("Login unsuccessful")


@click.command(help="Logout from CentML")
def logout():
    auth.remove_centml_cred()
    click.echo("Logout successful")
