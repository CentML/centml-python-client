import sys
from functools import wraps
from typing import Dict
import click
from tabulate import tabulate
from centml.sdk import DeploymentType, DeploymentStatus, ServiceStatus, ApiException, HardwareInstanceResponse
from centml.sdk.api import get_centml_client


depl_name_to_type_map = {
    "inference": DeploymentType.INFERENCE_V2,
    "compute": DeploymentType.COMPUTE_V2,
    "cserve": DeploymentType.CSERVE,
}
depl_type_to_name_map = {v: k for k, v in depl_name_to_type_map.items()}


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
    return ssh_key[:10] + '...'


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
        rows = [
            [d.id, d.name, depl_type_to_name_map[d.type], d.status.value, d.created_at.strftime("%Y-%m-%d %H:%M:%S")]
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
@click.argument("name", type=str)
@handle_exception
def get(name):
    with get_centml_client() as cclient:
        # Retrieve all deployments and search for the given name
        deployments = cclient.get(None)
        deployment = next((d for d in deployments if d.name == name), None)

        if deployment is None:
            sys.exit(f"Deployment with name '{name}' not found.")

        depl_type = deployment.type
        depl_id = deployment.id

        # Now retrieve the full deployment details based on the type
        if depl_type == DeploymentType.INFERENCE_V2:
            deployment = cclient.get_inference(depl_id)
        elif depl_type == DeploymentType.COMPUTE_V2:
            deployment = cclient.get_compute(depl_id)
        elif depl_type == DeploymentType.CSERVE:
            deployment = cclient.get_cserve(depl_id)
        else:
            sys.exit("Unknown deployment type.")

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
                    ("Cost", f"{hw.cost_per_hr/100} credits/hr"),
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
                        ("Container port", deployment.port),
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
        elif depl_type == DeploymentType.CSERVE:
            click.echo(
                tabulate(
                    [
                        ("Hugging face model", deployment.model),
                        (
                            "Parallelism",
                            {"tensor": deployment.tensor_parallel_size, "pipeline": deployment.pipeline_parallel_size},
                        ),
                        ("Replicas", {"min": deployment.min_scale, "max": deployment.max_scale}),
                        ("Max concurrency", deployment.concurrency or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )


@click.command(help="Create a new deployment")
@handle_exception
def create():
    with get_centml_client() as cclient:
        # Prompt for general fields
        name = click.prompt("Enter a name for the deployment")
        dtype_str = click.prompt(
            "Select a deployment type",
            type=click.Choice(list(depl_name_to_type_map.keys())),
            show_choices=True
        )
        depl_type = depl_name_to_type_map[dtype_str]

        # Select cluster
        clusters = cclient.get_clusters().results
        if not clusters:
            click.echo("No clusters available. Please ensure you have a cluster setup.")
            return
        cluster_names = [c.display_name for c in clusters]
        cluster_name = click.prompt(
            "Select a cluster",
            type=click.Choice(cluster_names),
            show_choices=True
        )
        cluster_id = next(c.id for c in clusters if c.display_name == cluster_name)

        # Hardware selection
        hw_resp = cclient.get_hardware_instances(cluster_id)
        if not hw_resp:
            click.echo("No hardware instances available for this cluster.")
            return
        hw_names = [h.name for h in hw_resp]
        hw_name = click.prompt(
            "Select a hardware instance",
            type=click.Choice(hw_names),
            show_choices=True
        )
        hw_id = next(h.id for h in hw_resp if h.name == hw_name)

        # Common fields
        min_scale = click.prompt("Minimum number of replicas", default=1, type=int)
        max_scale = click.prompt("Maximum number of replicas", default=1, type=int)
        concurrency = click.prompt("Max concurrency (or leave blank)", default="", show_default=False)
        concurrency = int(concurrency) if concurrency else None

        # Depending on type:
        if depl_type == DeploymentType.INFERENCE_V2:
            image = click.prompt("Enter the image URL")
            port = click.prompt("Enter the container port", default=8080, type=int)
            healthcheck = click.prompt("Enter healthcheck endpoint (default '/')", default="/", show_default=True)
            env_vars_str = click.prompt("Enter environment variables in KEY=VALUE format (comma separated) or leave blank", default="", show_default=False)
            env_vars = {}
            if env_vars_str.strip():
                for kv in env_vars_str.split(","):
                    k, v = kv.strip().split("=")
                    env_vars[k] = v

            # Construct the inference request
            from platform_api_python_client import CreateInferenceDeploymentRequest
            req = CreateInferenceDeploymentRequest(
                name=name,
                cluster_id=cluster_id,
                hardware_instance_id=hw_id,
                image_url=image,
                port=port,
                healthcheck=healthcheck,
                min_scale=min_scale,
                max_scale=max_scale,
                concurrency=concurrency,
                env_vars=env_vars if env_vars else None
            )
            created = cclient.create_inference(req)
            click.echo(f"Inference deployment {name} created with ID: {created.id}")

        elif depl_type == DeploymentType.COMPUTE_V2:
            # For compute deployments, we might ask for a public SSH key
            ssh_key = click.prompt("Enter your public SSH key", default="", show_default=False)

            from platform_api_python_client import CreateComputeDeploymentRequest
            req = CreateComputeDeploymentRequest(
                name=name,
                cluster_id=cluster_id,
                hardware_instance_id=hw_id,
                ssh_public_key=ssh_key if ssh_key.strip() else None
            )
            created = cclient.create_compute(req)
            click.echo(f"Compute deployment {name} created with ID: {created.id}")

        elif depl_type == DeploymentType.CSERVE:
            # For cserve deployments, ask for model and parallelism
            model = click.prompt("Enter the Hugging Face model", default="facebook/opt-1.3b")
            tensor_parallel_size = click.prompt("Tensor parallel size", default=1, type=int)
            pipeline_parallel_size = click.prompt("Pipeline parallel size", default=1, type=int)
            # concurrency asked above

            from platform_api_python_client import CreateCServeDeploymentRequest
            req = CreateCServeDeploymentRequest(
                name=name,
                cluster_id=cluster_id,
                hardware_instance_id=hw_id,
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                min_scale=min_scale,
                max_scale=max_scale,
                concurrency=concurrency
            )
            created = cclient.create_cserve(req)
            click.echo(f"CServe deployment {name} created with ID: {created.id}")

        else:
            click.echo("Unknown deployment type.")


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
