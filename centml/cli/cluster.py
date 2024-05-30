import click
from tabulate import tabulate
import platform_api_client
from platform_api_client.models.endpoint_ready_state import EndpointReadyState
from platform_api_client.models.deployment_status import DeploymentStatus

from ..sdk import api


hw_to_id_map = {"small": 1000, "medium": 1001, "large": 1002}
id_to_hw_map = {v: k for k, v in hw_to_id_map.items()}


depl_type_map = {
    "inference": platform_api_client.DeploymentType.INFERENCE,
    "compute": platform_api_client.DeploymentType.COMPUTE,
}


def get_ready_status(api_status, service_status):
    if api_status == DeploymentStatus.PAUSED:
        return click.style("paused", fg="yellow")
    elif api_status == DeploymentStatus.DELETED:
        return click.style("deleted", fg="white")
    elif api_status == DeploymentStatus.FAILED:
        return click.style("failed", fg="red")
    elif api_status == DeploymentStatus.ACTIVE and service_status == EndpointReadyState.NUMBER_1:
        return click.style("ready", fg="green")
    elif api_status == DeploymentStatus.ACTIVE and service_status == EndpointReadyState.NUMBER_2:
        return click.style("starting", fg="cyan")
    else:
        return click.style("unknown", fg="black", bg="white")


@click.command(help="List all deployments")
@click.argument("type", default="all")
def ls(type):
    depl_type = depl_type_map[type] if type in depl_type_map else None
    rows = api.get(depl_type)

    click.echo(
        tabulate(
            rows,
            headers=["ID", "Name", "Type", "Status", "Created at"],
            tablefmt="rounded_outline",
            disable_numparse=True,
        )
    )


@click.command(help="Get deployment details")
@click.argument("id", type=int)
def get(id):
    deployment = api.get_inference(id)
    state = api.get_status(id)
    ready_status = get_ready_status(deployment.status, state.service_status)

    click.echo(f"Inference deployment #{id} is {ready_status}")
    click.echo(
        tabulate(
            [
                ("Name", deployment.name),
                ("Image", deployment.image_url),
                ("Endpoint", f"https://{deployment.endpoint_url}/"),
                ("Created at", deployment.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                ("Hardware", id_to_hw_map[deployment.hardware_instance_id]),
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
                ("Hardware", id_to_hw_map[deployment.hardware_instance_id]),
                ("Port", deployment.port),
                ("Healthcheck", deployment.healthcheck or "/"),
                ("Replicas", {"min": deployment.min_replicas, "max": deployment.max_replicas}),
                ("Environment variables", deployment.env_vars or "None"),
            ],
            tablefmt="rounded_outline",
            disable_numparse=True,
        )
    )


@click.command(help="Create a new deployment")
@click.argument("type", type=click.Choice(list(depl_type_map.keys())))
@click.option("--name", "-n", prompt="Name", help="Name of the deployment")
@click.option("--image", "-i", prompt="Image", help="Container image")
@click.option("--port", "-p", prompt="Port", type=int, help="Port to expose")
@click.option(
    "--hardware", "-h", prompt="Hardware", type=click.Choice(list(hw_to_id_map.keys())), help="Hardware instance type"
)
@click.option("--health", default="/", prompt="Health check", help="Health check endpoint")
@click.option("--min_replicas", default="1", prompt="Min replicas", type=click.IntRange(1, 10))
@click.option("--max_replicas", default="1", prompt="Max replicas", type=click.IntRange(1, 10))
@click.option("--username", prompt=True, default="", help="Username for HTTP authentication")
@click.option("--password", prompt=True, default="", hide_input=True, help="Password for HTTP authentication")
@click.option("--env", "-e", required=False, type=str, multiple=True, help="Environment variables (KEY=VALUE)")
def create(type, name, image, port, hardware, health, min_replicas, max_replicas, username, password, env):
    resp = api.create_inference(
        name, image, port, hw_to_id_map[hardware], health, min_replicas, max_replicas, username, password, env
    )
    click.echo(f"Inference deployment #{resp.id} created at https://{resp.endpoint_url}/")


@click.command(help="Delete a deployment")
@click.argument("id", type=int)
def delete(id):
    api.delete(id)


@click.command(help="Pause a deployment")
@click.argument("id", type=int)
def pause(id):
    api.pause(id)


@click.command(help="Resume a deployment")
@click.argument("id", type=int)
def resume(id):
    api.resume(id)
