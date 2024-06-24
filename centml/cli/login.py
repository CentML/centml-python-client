import click

from centml.sdk import auth, config


@click.command(help="Login to CentML")
@click.argument("token_file", required=False)
def login(token_file):
    if token_file:
        auth.store_centml_cred(token_file)

    if auth.load_centml_cred():
        click.echo(f"Authenticating with credentials from {config.Config.centml_cred_file}\n")
        click.echo("Login successful")
    else:
        click.echo("Login with CentML authentication token")
        click.echo("Usage: centml login TOKEN_FILE\n")
        choice = click.confirm("Do you want to download the token?")

        if choice:
            click.launch(f"{config.Config.centml_web_url}?isCliAuthenticated=true")
        else:
            click.echo("Login unsuccessful")


@click.command(help="Logout from CentML")
def logout():
    auth.remove_centml_cred()
    click.echo("Logout successful")
