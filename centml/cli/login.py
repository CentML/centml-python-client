import click

from centml.sdk import auth
from centml.sdk.config import settings


@click.command(help="Login to CentML")
@click.argument("token_file", required=False)
def login(token_file):
    if token_file:
        auth.store_centml_cred(token_file)

    if auth.load_centml_cred():
        click.echo(f"Authenticating with credentials from {settings.CENTML_CRED_FILE_PATH}\n")
        click.echo("Login successful")
    else:
        click.echo("Login with CentML authentication token")
        click.echo("Usage: centml login TOKEN_FILE\n")
        choice = click.confirm("Do you want to download the token?")

        if choice:
            click.launch(f"{settings.CENTML_WEB_URL}?isCliAuthenticated=true")
        else:
            click.echo("Login unsuccessful")


@click.command(help="Logout from CentML")
def logout():
    auth.remove_centml_cred()
    click.echo("Logout successful")
