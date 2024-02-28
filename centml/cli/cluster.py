import contextlib
import click
from tabulate import tabulate
import platform_api_client
from platform_api_client.models.endpoint_ready_state import EndpointReadyState
from platform_api_client.models.deployment_status import DeploymentStatus

from . import login
from .config import Config


hw_to_id_map = {
    "small": 1000,
    "medium": 1001,
    "large": 1002,
}
id_to_hw_map = {v: k for k, v in hw_to_id_map.items()}


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


@contextlib.contextmanager
def get_api():
    configuration = platform_api_client.Configuration(
        host=Config.platformapi_url, access_token=login.get_centml_token()
    )

    with platform_api_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_client.EXTERNALApi(api_client)

        yield api_instance


@click.command()
def test():
    with get_api() as api:
        resp = api.get_hardware_instances_hardware_instances_get()
        print(resp)

@click.command(help="List all deployments")
def ls():
    with get_api() as api:
        results = api.get_deployments_deployments_get().results
        deployments = sorted(results, reverse=True, key=lambda d: d.created_at)

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
@click.argument("id", type=int)
def get(id):
    with get_api() as api:
        deployment = api.get_inference_deployment_deployments_inference_deployment_id_get(id)
        state = api.get_deployment_status_deployments_status_deployment_id_get(id)
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
@click.option("--name", "-n", prompt="Name", help="Name of the deployment")
@click.option("--image", "-i", prompt="Image", help="Container image")
@click.option("--port", "-p", prompt="Port", type=int, help="Port to expose")
@click.option("--hardware", "-h", prompt="Hardware", type=click.Choice(hw_to_id_map.keys()),
    help="Hardware instance type")
@click.option("--health", default="/", prompt="Health check", help="Health check endpoint")
@click.option("--min_replicas", default="1", prompt="Min replicas", type=click.IntRange(1, 10))
@click.option("--max_replicas", default="1", prompt="Max replicas", type=click.IntRange(1, 10))
@click.option("--username", prompt=True, default="",
    help="Username for HTTP authentication")
@click.option("--password", prompt=True, default="", hide_input=True,
    help="Password for HTTP authentication")
@click.option("--env", "-e", required=False, type=str, multiple=True,
    help="Environment variables (KEY=VALUE)")
def create(name, image, port, hardware, health, min_replicas, max_replicas, username, password, env):
    with get_api() as api:
        req = platform_api_client.CreateInferenceDeploymentRequest(
            name=name,
            image_url=image,
            hardware_instance_id=hw_to_id_map[hardware],
            env_vars={k:v for (k,v) in env},
            secrets=platform_api_client.AuthSecret(
                username=username,
                password=password,
            ) if username and password else None,
            port=port,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            timeout=0,
            healthcheck=health,
        )
        resp = api.create_inference_deployment_deployments_inference_post(req)
        click.echo(f"Inference deployment #{resp.id} created at https://{resp.endpoint_url}/")


def update_status(id, new_status):
    with get_api() as api:
        status_req = platform_api_client.DeploymentStatusRequest(status=new_status)
        api.update_deployment_status_deployments_status_deployment_id_put(id, status_req)


@click.command(help="Delete a deployment")
@click.argument("id", type=int)
def delete(id):
    update_status(id, DeploymentStatus.DELETED)


@click.command(help="Pause a deployment")
@click.argument("id", type=int)
def pause(id):
    update_status(id, DeploymentStatus.PAUSED)


@click.command(help="Resume a deployment")
@click.argument("id", type=int)
def resume(id):
    update_status(id, DeploymentStatus.ACTIVE)
