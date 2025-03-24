import sys
from functools import wraps
from typing import Dict
import click
from tabulate import tabulate
from centml.sdk import DeploymentType, DeploymentStatus, ServiceStatus, ApiException, HardwareInstanceResponse
from centml.sdk.api import get_centml_client


depl_type_to_name_map = {
    DeploymentType.INFERENCE: 'inference',
    DeploymentType.COMPUTE: 'compute',
    DeploymentType.COMPILATION: 'compilation',
    DeploymentType.INFERENCE_V2: 'inference',
    DeploymentType.COMPUTE_V2: 'compute',
    DeploymentType.CSERVE: 'cserve',
    DeploymentType.CSERVE_V2: 'cserve',
    DeploymentType.RAG: 'rag',
}
depl_name_to_type_map = {
    'inference': DeploymentType.INFERENCE_V2,
    'cserve': DeploymentType.CSERVE_V2,
    'compute': DeploymentType.COMPUTE_V2,
    'rag': DeploymentType.RAG,
}


def handle_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            click.echo(f"Error: {e.body or e.reason}")
            return None

    return wrapper


def _get_hw_to_id_map(cclient, cluster_id):
    response = cclient.get_hardware_instances(cluster_id)

    # Initialize hashmap for hardware to id or vice versa mapping
    hw_to_id_map: Dict[str, int] = {}
    id_to_hw_map: Dict[int, HardwareInstanceResponse] = {}

    for hw in response:
        hw_to_id_map[hw.name] = hw.id
        id_to_hw_map[hw.id] = hw
    return hw_to_id_map, id_to_hw_map


def _format_ssh_key(ssh_key):
    if not ssh_key:
        return "No SSH Key Found"
    return ssh_key[:32] + "..."


def _get_ready_status(cclient, deployment):
    api_status = deployment.status
    service_status = (
        cclient.get_status(deployment.id).service_status if deployment.status == DeploymentStatus.ACTIVE else None
    )

    status_styles = {
        (DeploymentStatus.PAUSED, None): ("paused", "yellow", "black"),
        (DeploymentStatus.DELETED, None): ("deleted", "white", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.HEALTHY): ("ready", "green", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.INITIALIZING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.MISSING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.ERROR): ("error", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.CREATECONTAINERCONFIGERROR): (
            "createContainerConfigError",
            "red",
            "black",
        ),
        (DeploymentStatus.ACTIVE, ServiceStatus.CRASHLOOPBACKOFF): ("crashLoopBackOff", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.IMAGEPULLBACKOFF): ("imagePullBackOff", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.PROGRESSDEADLINEEXCEEDED): ("progressDeadlineExceeded", "red", "black"),
    }

    style = status_styles.get((api_status, service_status), ("unknown", "black", "white"))
    # Handle foreground and background colors
    return click.style(style[0], fg=style[1], bg=style[2])


@click.command(help="List all deployments")
@click.argument("type", type=click.Choice(list(depl_name_to_type_map.keys())), required=False, default=None)
def ls(type):
    with get_centml_client() as cclient:
        depl_type = depl_name_to_type_map[type] if type in depl_name_to_type_map else None
        deployments = cclient.get(depl_type)
        rows = []
        for d in deployments:
            if d.type in depl_type_to_name_map:
                rows.append(
                    [
                        d.id,
                        d.name,
                        depl_type_to_name_map[d.type],
                        d.status.value,
                        d.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )

        click.echo(
            tabulate(
                rows,
                headers=["ID", "Name", "Type", "Status", "Created at"],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )


@click.command(help="Get deployment details")
@click.argument("type", type=click.Choice(list(depl_name_to_type_map.keys())))
@click.argument("id", type=int)
@handle_exception
def get(type, id):
    with get_centml_client() as cclient:
        depl_type = depl_name_to_type_map[type]

        if depl_type == DeploymentType.INFERENCE_V2:
            deployment = cclient.get_inference(id)
        elif depl_type == DeploymentType.COMPUTE_V2:
            deployment = cclient.get_compute(id)
        elif depl_type == DeploymentType.CSERVE_V2:
            deployment = cclient.get_cserve(id)
        else:
            sys.exit("Please enter correct deployment type")

        ready_status = _get_ready_status(cclient, deployment)
        _, id_to_hw_map = _get_hw_to_id_map(cclient, deployment.cluster_id)
        hw = id_to_hw_map[deployment.hardware_instance_id]

        click.echo(
            tabulate(
                [
                    ("Name", deployment.name),
                    ("Status", ready_status),
                    ("Endpoint", deployment.endpoint_url),
                    ("Created at", deployment.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                    ("Hardware", f"{hw.name} ({hw.num_gpu}x {hw.gpu_type})"),
                    ("Cost", f"{hw.cost_per_hr / 100} credits/hr"),
                ],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )

        click.echo("Additional deployment configurations:")
        if depl_type == DeploymentType.INFERENCE_V2:
            click.echo(
                tabulate(
                    [
                        ("Image", deployment.image_url),
                        ("Container port", deployment.container_port),
                        ("Healthcheck", deployment.healthcheck or "/"),
                        ("Replicas", {"min": deployment.min_scale, "max": deployment.max_scale}),
                        ("Environment variables", deployment.env_vars or "None"),
                        ("Max concurrency", deployment.concurrency or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )
        elif depl_type == DeploymentType.COMPUTE_V2:
            click.echo(
                tabulate(
                    [("Username", "centml"), ("SSH key", _format_ssh_key(deployment.ssh_public_key))],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )
        elif depl_type == DeploymentType.CSERVE_V2:
            click.echo(
                tabulate(
                    [
                        ("Hugging face model", deployment.recipe.model),
                        (
                            "Parallelism",
                            {
                                "tensor": deployment.recipe.additional_properties['tensor_parallel_size'],
                                "pipeline": deployment.recipe.additional_properties['pipeline_parallel_size'],
                            },
                        ),
                        ("Replicas", {"min": deployment.min_scale, "max": deployment.max_scale}),
                        ("Max concurrency", deployment.concurrency or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )


@click.command(help="Delete a deployment")
@click.argument("id", type=int)
@handle_exception
def delete(id):
    with get_centml_client() as cclient:
        cclient.delete(id)
        click.echo("Deployment has been deleted")


@click.command(help="Pause a deployment")
@click.argument("id", type=int)
@handle_exception
def pause(id):
    with get_centml_client() as cclient:
        cclient.pause(id)
        click.echo("Deployment has been paused")


@click.command(help="Resume a deployment")
@click.argument("id", type=int)
@handle_exception
def resume(id):
    with get_centml_client() as cclient:
        cclient.resume(id)
        click.echo("Deployment has been resumed")
