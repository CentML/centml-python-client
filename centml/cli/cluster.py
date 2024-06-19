import sys
from typing import Dict
import click
from tabulate import tabulate
import platform_api_client
from platform_api_client.models.endpoint_ready_state import EndpointReadyState
from platform_api_client.models.deployment_status import DeploymentStatus
from centml.sdk import api


# Custom class to parse key-value pairs for env variables for inference deployment
class InferenceEnvType(click.ParamType):
    name = "key_value"

    def convert(self, value, param, ctx):
        try:
            key, val = value.split('=', 1)
            return key, val
        except ValueError:
            self.fail(f"{value} is not a valid key=value pair", param, ctx)
            return None  # to avoid warning from lint for inconsistent return statements


def get_hw_to_id_map():
    response = api.get_hardware_instances()

    # Convert to list of dictionaries
    instances = [item.to_dict() for item in response]

    # Initialize hashmap for hardware to id or vice versa mapping
    hw_to_id_map: Dict[str, int] = {}
    id_to_hw_map: Dict[str, str] = {}

    for item in instances:
        hw_to_id_map[item["name"]] = item["id"]
        id_to_hw_map[item["id"]] = item["name"]
    return hw_to_id_map, id_to_hw_map


depl_type_map = {
    "inference": platform_api_client.DeploymentType.INFERENCE,
    "compute": platform_api_client.DeploymentType.COMPUTE,
}


def format_ssh_key(ssh_key):
    if not ssh_key:
        return "No SSH Key Found"
    return ssh_key[:10] + '...'


def get_ready_status(api_status, service_status):
    status_styles = {
        (DeploymentStatus.PAUSED, None): ("paused", "yellow", "black"),
        (DeploymentStatus.DELETED, None): ("deleted", "white", "black"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.READY): ("ready", "green", "black"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.NOT_READY): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.NOT_FOUND): ("not found", "cyan"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.FOUND_MULTIPLE): ("found multiple", "black", "white"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.INGRESS_RULE_NOT_FOUND): (
            "ingress rule not found",
            "black",
            "white",
        ),
        (DeploymentStatus.ACTIVE, EndpointReadyState.CONDITION_NOT_FOUND): ("condition not found", "black", "white"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.INGRESS_NOT_CONFIGURED): (
            "ingress not configured",
            "black",
            "white",
        ),
        (DeploymentStatus.ACTIVE, EndpointReadyState.CONTAINER_MISSING): ("container missing", "black", "white"),
        (DeploymentStatus.ACTIVE, EndpointReadyState.PROGRESS_DEADLINE_EXCEEDED): (
            "progress deadline exceeded",
            "black",
            "white",
        ),
        (DeploymentStatus.ACTIVE, EndpointReadyState.REVISION_MISSING): ("revision missing", "black", "white"),
    }

    style = status_styles.get((api_status, service_status), ("unknown", "black", "white"))
    # Handle foreground and background colors
    return click.style(style[0], fg=style[1], bg=style[2])


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
@click.argument("type", type=click.Choice(list(depl_type_map.keys())))
@click.argument("id", type=int)
def get(type, id):
    if type == platform_api_client.DeploymentType.INFERENCE:
        deployment = api.get_inference(id)
    elif type == platform_api_client.DeploymentType.COMPUTE:
        deployment = api.get_compute(id)
    else:
        sys.exit("Please enter correct deployment type")
    state = api.get_status(id)
    ready_status = get_ready_status(deployment.status, state.service_status)

    click.echo(f"The current status of Deployment #{id} is: {ready_status}.")

    _, id_to_hw_map = get_hw_to_id_map()

    click.echo(
        tabulate(
            [
                ("Name", deployment.name),
                ("Image", deployment.image_url),
                ("Endpoint", deployment.endpoint_url),
                ("Created at", deployment.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                ("Hardware", id_to_hw_map[deployment.hardware_instance_id]),
            ],
            tablefmt="rounded_outline",
            disable_numparse=True,
        )
    )

    click.echo("Additional deployment configurations:")
    if type == platform_api_client.DeploymentType.INFERENCE:
        click.echo(
            tabulate(
                [
                    ("Port", deployment.port),
                    ("Healthcheck", deployment.healthcheck or "/"),
                    ("Replicas", {"min": deployment.min_replicas, "max": deployment.max_replicas}),
                    ("Environment variables", deployment.env_vars or "None"),
                    ("Max concurrency", deployment.timeout or "None"),
                ],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )
    elif type == platform_api_client.DeploymentType.COMPUTE:
        click.echo(
            tabulate(
                [
                    ("Port", deployment.port),
                    ("Username", deployment.username or "None"),
                    ("SSH key", format_ssh_key(deployment.ssh_key)),
                ],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )


# Define common deployment
def common_options(func):
    hw_to_id_map, _ = get_hw_to_id_map()
    func = click.option("--name", "-n", prompt="Name", help="Name of the deployment")(func)
    func = click.option("--image", "-i", prompt="Image", help="Container image")(func)
    func = click.option(
        "--hardware",
        "-h",
        prompt="Hardware",
        type=click.Choice(list(hw_to_id_map.keys())),
        help="Hardware instance type",
    )(func)
    return func


# Define inference specific options
def inference_options(func):
    func = click.option("--port", "-p", prompt="Port", type=int, help="Port to expose")(func)
    func = click.option(
        "--env", type=InferenceEnvType(), help="Environment variables in the format KEY=VALUE", multiple=True
    )(func)
    func = click.option("--min_replicas", default="1", prompt="Min replicas", type=click.IntRange(1, 10))(func)
    func = click.option("--max_replicas", default="1", prompt="Max replicas", type=click.IntRange(1, 10))(func)
    func = click.option("--health", default="/", prompt="Health check", help="Health check endpoint")(func)
    func = click.option("--is_private", default=False, type=bool, prompt="Is private?", help="Is private endpoint?")(
        func
    )
    func = click.option("--timeout", prompt="Max concurrency", default=0, type=int)(func)
    func = click.option("--command", type=str, required=False, default=None, help="Define a command for a container")(
        func
    )
    func = click.option("--command_args", multiple=True, type=str, default=None, help="List of command arguments")(func)
    return func


# Define compute specific options
def compute_options(func):
    func = click.option("--username", prompt="Username", type=str, help="Username")(func)
    func = click.option("--password", prompt="Password", hide_input=True, type=str, help="password")(func)
    func = click.option(
        "--ssh_key", prompt="Add ssh key", default="", type=str, help="Would you like to add an SSH key?"
    )(func)
    return func


# Main command group
@click.group(help="Create a new deployment")
@click.pass_context
def create(ctx):
    pass


# Define the inference subcommand
@create.command(name="inference", help="Create an inference deployment")
@common_options
@inference_options
@click.pass_context
def create_inference(ctx, **kwargs):
    click.echo("Creating inference deployment with the following options:")

    name = kwargs.get("name")
    image = kwargs.get("image")
    port = kwargs.get("port")
    is_private = kwargs.get("is_private")
    hardware = kwargs.get("hardware")
    health = kwargs.get("health")
    min_replicas = kwargs.get("min_replicas")
    max_replicas = kwargs.get("max_replicas")
    env = kwargs.get("env")
    command = kwargs.get("command")
    command_args = kwargs.get("command_args")
    timeout = kwargs.get("timeout")

    hw_to_id_map, _ = get_hw_to_id_map()

    # Call the API function for creating infrence deployment
    resp = api.create_inference(
        name,
        image,
        port,
        is_private,
        hw_to_id_map[hardware],
        health,
        min_replicas,
        max_replicas,
        env,
        command,
        command_args,
        timeout,
    )

    click.echo(f"Inference deployment #{resp.id} created at https://{resp.endpoint_url}/")


# Define the compute subcommand
@create.command(name="compute", help="Create a compute deployment")
@common_options
@compute_options
@click.pass_context
def create_compute(ctx, **kwargs):
    click.echo("Creating compute deployment with the following options:")

    name = kwargs.get("name")
    image = kwargs.get("image")
    username = kwargs.get("username")
    password = kwargs.get("password")
    ssh_key = kwargs.get("ssh_key")
    hardware = kwargs.get("hardware")

    hw_to_id_map, _ = get_hw_to_id_map()

    # Call the API function for creating infrence deployment
    resp = api.create_compute(name, image, username, password, ssh_key, hw_to_id_map[hardware])

    click.echo(f"Compute deployment #{resp.id} created at https://{resp.endpoint_url}/")


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
