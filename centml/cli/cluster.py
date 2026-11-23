import sys
from functools import wraps
from typing import Dict
import click
import yaml
from pydantic import ValidationError
from tabulate import tabulate
from centml.sdk import (
    DeploymentType,
    DeploymentStatus,
    ServiceStatus,
    RolloutStatus,
    ApiException,
    HardwareInstanceResponse,
    CloudProvider,
    ClusterCredential,
    ClusterRegistrationRequest,
    ClusterBaseComponents,
    UpdateClusterRequest,
)
from centml.sdk.api import get_centml_client

# convert deployment type enum to a user friendly name
depl_type_to_name_map = {
    DeploymentType.INFERENCE: "inference",
    DeploymentType.COMPUTE: "compute",
    DeploymentType.COMPILATION: "compilation",
    DeploymentType.INFERENCE_V2: "inference",
    DeploymentType.INFERENCE_V3: "inference",
    DeploymentType.COMPUTE_V2: "compute",
    DeploymentType.RAG: "rag",
    DeploymentType.JOB: "job",
    DeploymentType.DYNAMO: "dynamo",
}
# use latest type to for user requests
depl_name_to_type_map = {
    "inference": DeploymentType.INFERENCE_V3,
    "compute": DeploymentType.COMPUTE_V2,
    "rag": DeploymentType.RAG,
    "job": DeploymentType.JOB,
}
rollout_status_to_service_status_map = {
    RolloutStatus.HEALTHY: ServiceStatus.HEALTHY,
    RolloutStatus.MISSING: ServiceStatus.MISSING,
    RolloutStatus.PROGRESSING: ServiceStatus.INITIALIZING,
    RolloutStatus.DEGRADED: ServiceStatus.ERROR,
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


def _get_replica_info(deployment):
    """Extract replica information handling V2/V3 field differences"""
    # Check actual deployment object fields rather than depl_type
    if hasattr(deployment, 'min_replicas'):
        # V3 deployment response object
        return {"min": deployment.min_replicas, "max": deployment.max_replicas}
    elif hasattr(deployment, 'min_scale'):
        # V2 deployment response object
        return {"min": deployment.min_scale, "max": deployment.max_scale}
    else:
        # Fallback - shouldn't happen
        return {"min": "N/A", "max": "N/A"}


def _get_ready_status(deployment, service_status):
    api_status = deployment.status

    status_styles = {
        (DeploymentStatus.PAUSED, None): ("paused", "yellow", "black"),
        (DeploymentStatus.DELETED, None): ("deleted", "white", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.HEALTHY): ("ready", "green", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.SCALINGUP): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.PULLING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.INITIALIZING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.MISSING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.NOTREADY): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.COMPLETED): ("completed", "green", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.CLEANEDUP): ("cleanedUp", "white", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.FAILED): ("failed", "red", "black"),
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


def _get_service_status(status_response, revision_number):
    if status_response is None:
        return None

    service_status = getattr(status_response, "service_status", None)
    if service_status is not None:
        return service_status

    revision_pod_details_list = getattr(status_response, "revision_pod_details_list", None) or []
    current_revision = next(
        (
            revision
            for revision in revision_pod_details_list
            if getattr(revision, "revision_number", None) == revision_number
        ),
        (
            revision_pod_details_list[0]
            if revision_pod_details_list and getattr(revision_pod_details_list[0], "revision_number") is None
            else None
        ),
    )
    revision_status = getattr(current_revision, "revision_status", None)

    return revision_status or rollout_status_to_service_status_map.get(getattr(status_response, "rollout_status", None))


def _append_status_error_message(messages, seen_messages, label, error_message):
    if not error_message or error_message in seen_messages:
        return

    seen_messages.add(error_message)
    messages.append(f"{label}: {error_message}")


def _get_status_error_messages(status_response):
    if status_response is None:
        return []

    error_message = getattr(status_response, "error_message", None)
    if error_message:
        return [error_message]

    messages = []
    seen_messages = set()

    for revision in getattr(status_response, "revision_pod_details_list", None) or []:
        revision_label = f"revision {revision.revision_number}" if revision.revision_number is not None else "revision"
        _append_status_error_message(messages, seen_messages, revision_label, revision.error_message)

        for pod in getattr(revision, "pod_details_list", None) or []:
            pod_label = pod.name or "pod"
            _append_status_error_message(messages, seen_messages, f"{revision_label} / {pod_label}", pod.error_message)

    return messages


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

        if depl_type in [DeploymentType.INFERENCE_V2, DeploymentType.INFERENCE_V3]:
            deployment = cclient.get_inference(id)  # handles both V2 and V3
        elif depl_type == DeploymentType.COMPUTE_V2:
            deployment = cclient.get_compute(id)
        elif depl_type == DeploymentType.JOB:
            deployment = cclient.get_job(id)
        else:
            sys.exit("Please enter correct deployment type")

        deployment_status = cclient.get_status(deployment.id) if deployment.status == DeploymentStatus.ACTIVE else None
        revision_number = getattr(deployment, "revision_number", None)
        service_status = _get_service_status(deployment_status, revision_number)
        ready_status = _get_ready_status(deployment, service_status)
        status_error_messages = _get_status_error_messages(deployment_status)
        _, id_to_hw_map = _get_hw_to_id_map(cclient, deployment.cluster_id)
        hw = id_to_hw_map[deployment.hardware_instance_id]
        detail_rows = [
            ("Name", deployment.name),
            ("Status", ready_status),
            ("Created at", deployment.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            ("Hardware", f"{hw.name} ({hw.num_gpu}x {hw.gpu_type})"),
        ]
        if depl_type != DeploymentType.JOB:
            detail_rows.insert(2, ("Endpoint", deployment.endpoint_url))

        click.echo(tabulate(detail_rows, tablefmt="rounded_outline", disable_numparse=True))
        if status_error_messages:
            click.echo("\nStatus errors:")
            for message in status_error_messages:
                click.echo(f"- {message}")

        click.echo("Additional deployment configurations:")
        if depl_type in [DeploymentType.INFERENCE_V2, DeploymentType.INFERENCE_V3]:
            replica_info = _get_replica_info(deployment)
            display_rows = [
                ("Image", deployment.image_url),
                ("Container port", deployment.container_port),
                ("Healthcheck", deployment.healthcheck or "/"),
                ("Replicas", replica_info),
                ("Environment variables", deployment.env_vars or "None"),
                ("Max concurrency", deployment.concurrency or "None"),
            ]

            click.echo(tabulate(display_rows, tablefmt="rounded_outline", disable_numparse=True))
        elif depl_type == DeploymentType.COMPUTE_V2:
            click.echo(
                tabulate(
                    [("Username", "centml"), ("SSH key", _format_ssh_key(deployment.ssh_public_key))],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )
        elif depl_type == DeploymentType.JOB:
            display_rows = [
                ("Image", deployment.image_url),
                ("Command", deployment.original_command or "None"),
                ("Environment variables", deployment.env_vars or "None"),
                ("Completions", deployment.completions),
                ("Parallelism", deployment.parallelism),
                ("Logging", deployment.enable_logging),
            ]

            click.echo(tabulate(display_rows, tablefmt="rounded_outline", disable_numparse=True))


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


@click.command(help="Show GPU capacity across clusters")
@click.option("--cluster-id", type=int, default=None, help="Filter to a specific cluster")
@handle_exception
def capacity(cluster_id):
    with get_centml_client() as cclient:
        clusters = cclient.get_capacity(cluster_id)

        if not clusters:
            click.echo("No accelerator capacity available")
            return

        rows = []
        for cluster in clusters:
            for gpu in cluster.gpu_types:
                utilization = (gpu.used_gpus / gpu.total_gpus * 100) if gpu.total_gpus > 0 else 0
                rows.append([cluster.cluster_name, gpu.gpu_type, gpu.used_gpus, gpu.total_gpus, f"{utilization:.1f}%"])

        click.echo(
            tabulate(
                rows,
                headers=["Cluster", "GPU Type", "Used", "Total", "Utilization"],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )


@click.command(help="Register your own Kubernetes cluster")
@click.option("--name", "display_name", required=True, help="Display name for the cluster")
@click.option("--storage-class", required=True, help="Storage class to use for deployments on the cluster")
@click.option("--server", required=True, help="URL of the cluster's Kubernetes API server")
@click.option(
    "--ca-cert",
    required=True,
    help="Base64-encoded PEM CA certificate of the API server (a kubeconfig's certificate-authority-data)",
)
@click.option(
    "--bearer-token",
    required=True,
    prompt=True,
    hide_input=True,
    help="Bearer token for the API server; prompted for when omitted",
)
@click.option("--region", default=None, help="Region the cluster runs in")
@click.option(
    "--cloud-provider",
    type=click.Choice([p.value for p in CloudProvider if p != CloudProvider.UNKNOWN]),
    default=None,
    help="Cloud provider the cluster runs on",
)
@handle_exception
# pylint: disable=R0917
def register(display_name, storage_class, server, ca_cert, bearer_token, region, cloud_provider):
    request = ClusterRegistrationRequest(
        display_name=display_name,
        storage_class=storage_class,
        region=region,
        cloud_provider=CloudProvider(cloud_provider) if cloud_provider else None,
        credential=ClusterCredential(server=server, bearer_token=bearer_token, ca_cert=ca_cert),
    )
    with get_centml_client() as cclient:
        response = cclient.register_cluster(request)
    click.echo(f"Registered cluster {response.id}")


@click.command(
    name="update_cluster",
    help="""Set the components installed on a registered cluster.

COMPONENTS_FILE is a YAML or JSON mapping of component name to its Helm chart values
({} for the defaults). final_stack is required. Components left out of the file are
uninstalled from the cluster.""",
)
@click.argument("cluster_id", type=int)
@click.argument("components_file", type=click.File("r"))
@handle_exception
def update_cluster(cluster_id, components_file):
    try:
        components = yaml.safe_load(components_file)
    except yaml.YAMLError as e:
        raise click.BadParameter(f"not valid YAML: {e}", param_hint="COMPONENTS_FILE") from e
    if not isinstance(components, dict):
        raise click.BadParameter("must be a mapping of component name to chart values", param_hint="COMPONENTS_FILE")
    # The generated model silently drops unknown names, which would uninstall a misspelled component.
    unknown = sorted(set(components) - set(ClusterBaseComponents.model_fields))
    if unknown:
        raise click.BadParameter(
            f"unknown component(s) {', '.join(unknown)}; expected any of "
            f"{', '.join(ClusterBaseComponents.model_fields)}",
            param_hint="COMPONENTS_FILE",
        )
    # from_dict, not the constructors: they drop chart keys the typed models don't list.
    try:
        request = UpdateClusterRequest.from_dict({"components": components})
    except ValidationError as e:
        raise click.BadParameter(str(e), param_hint="COMPONENTS_FILE") from e
    with get_centml_client() as cclient:
        cclient.update_cluster(cluster_id, request)
    click.echo(f"Updating components on cluster {cluster_id}")
