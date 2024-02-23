import click
import contextlib
import platform_api_client
from tabulate import tabulate

from . import login
from .config import Config


@contextlib.contextmanager
def get_api():
    configuration = platform_api_client.Configuration(
        host = Config.platformapi_url,
        access_token = login.get_centml_token(),
    )

    with platform_api_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_client.EXTERNALApi(api_client)

        yield api_instance


@click.command()
def ls():
    with get_api() as api:
        deployments = sorted(api.get_deployments_deployments_get().results,
            reverse = True, key = lambda d: d.created_at,
        )

        rows = [[
            d.id,
            d.name,
            d.type.value,
            d.status.value,
            d.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            d.hardware_instance_id,
        ] for d in deployments]

        click.echo(tabulate(
            rows,
            headers=["ID", "Name", "Type", "Status", "Created at", "Hardware"],
            tablefmt="rounded_outline",
            disable_numparse=True,
        ))


@click.command()
def deploy():
    click.echo("deploy")


@click.command()
def delete():
    click.echo("delete")


@click.command()
def status():
    click.echo("status")
