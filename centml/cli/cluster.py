import contextlib
import click
from tabulate import tabulate
import platform_api_client

from . import login
from .config import Config


def get_hw(id):
    match id:
        case 1000:
            return "small"
        case 1001:
            return "medium"
        case 1002:
            return "large"


@contextlib.contextmanager
def get_api():
    configuration = platform_api_client.Configuration(
        host=Config.platformapi_url, access_token=login.get_centml_token()
    )

    with platform_api_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_client.EXTERNALApi(api_client)

        yield api_instance


@click.command(help="List all deployments")
def ls():
    with get_api() as api:
        deployments = sorted(api.get_deployments_deployments_get().results, reverse=True, key=lambda d: d.created_at)

        rows = [
            [d.id, d.name, d.type.value, d.status.value, d.created_at.strftime("%Y-%m-%d %H:%M:%S")]
            for d in deployments
        ]

        click.echo(
            tabulate(
                rows,
                headers=["ID", "Name", "Type", "Status", "Created at"],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )


@click.command(help="Get deployment details")
@click.argument("type", type=click.Choice(["inference", "compute"], case_sensitive=False))
@click.argument("id", type=int)
def get(type, id):
    with get_api() as api:
        if type == "inference":
            deployment = api.get_inference_deployment_deployments_inference_deployment_id_get(id)

            click.echo(f"Inference deployment #{id} is {deployment.status.value}")
            click.echo(
                tabulate(
                    [
                        ("Name", deployment.name),
                        ("Image", deployment.image_url),
                        ("Endpoint", deployment.endpoint_url),
                        ("Created at", deployment.created_at),
                        ("Hardware", get_hw(deployment.hardware_instance_id)),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )

            click.echo("Additional deployment configurations:")
            click.echo(
                tabulate(
                    [
                        ("Is private?", deployment.secrets is not None),
                        ("Hardware", get_hw(deployment.hardware_instance_id)),
                        ("Port", deployment.port),
                        ("Healthcheck", deployment.healthcheck or "/"),
                        ("Replicas", {"min": deployment.min_replicas, "max": deployment.max_replicas}),
                        ("Timeout", deployment.timeout),
                        ("Environment variables", deployment.env_vars or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )


@click.command(help="Create a new deployment")
def create():
    click.echo("deploy")


@click.command(help="Delete a deployment")
def delete():
    click.echo("delete")
