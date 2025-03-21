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
    return ssh_key[:10] + "..."


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
            show_choices=True,
            default=list(depl_name_to_type_map.keys())[0],
        )
        depl_type = depl_name_to_type_map[dtype_str]

        if depl_type == DeploymentType.INFERENCE_V2:

            # Select cluster using a numbered list
            clusters = cclient.get_clusters().results
            if not clusters:
                click.echo("No clusters available. Please ensure you have a cluster setup.")
                return

            click.echo("Available clusters:")
            for idx, cluster in enumerate(clusters, start=1):
                click.echo(f"{idx}. {cluster.display_name}")
            cluster_choice = click.prompt("Select a cluster by number", type=int, default=1)
            selected_cluster = clusters[cluster_choice - 1]
            cluster_id = selected_cluster.id

            # Hardware selection using a numbered list
            hw_resp = cclient.get_hardware_instances(cluster_id)
            if not hw_resp:
                click.echo("No hardware instances available for this cluster.")
                return

            click.echo("Available hardware instances:")
            for idx, hw in enumerate(hw_resp, start=1):
                click.echo(f"{idx}. {hw.name}")
            hw_choice = click.prompt("Select a hardware instance by number", type=int, default=1)
            selected_hw = hw_resp[hw_choice - 1]
            hw_id = selected_hw.id

            # Retrieve prebuilt images for inference deployments
            prebuilt_images = cclient.get_prebuilt_images(depl_type=depl_type)

            # Build list of image labels
            image_choices = [img.label for img in prebuilt_images.results] if prebuilt_images.results else []

            # Right now we disable this other option to get a MVP out quickly.
            #image_choices.append("Other")

            chosen_label = click.prompt(
                "Select a prebuilt image label or choose 'Other' to provide a custom image URL",
                type=click.Choice(image_choices),
                show_choices=True,
                default=image_choices[0],
            )

            if chosen_label == "Other":
                image = click.prompt("Enter the custom image URL")
                port = click.prompt("Enter the container port for the image", default=8080, type=int)
                healthcheck = click.prompt(
                    "Enter healthcheck endpoint (default '/') for the image", default="/", show_default=True
                )
            else:
                # Find the prebuilt image with the matching label
                selected_prebuilt = next(img for img in prebuilt_images.results if img.label == chosen_label)
                # Prompt the user to select a tag from the available tags
                tag = click.prompt(
                    "Select a tag for the image",
                    type=click.Choice(selected_prebuilt.tags),
                    show_choices=True,
                    default=selected_prebuilt.tags[0],
                )
                # Combine the image URL with the chosen tag
                image = f"{selected_prebuilt.image_name}:{tag}"
                port = selected_prebuilt.port
                healthcheck = selected_prebuilt.healthcheck if selected_prebuilt.healthcheck else "/"


            env_vars_str = click.prompt(
                "Enter environment variables in KEY=VALUE format (comma separated) or leave blank",
                default="",
                show_default=False,
            )
            env_vars = {}
            if env_vars_str.strip():
                for kv in env_vars_str.split(","):
                    k, v = kv.strip().split("=")
                    env_vars[k] = v

            command_str = click.prompt(
                "Enter command (space-separated) or leave blank",
                default="",
                show_default=False
            )

            command = command_str.strip() if command_str.strip() else None

            # Common fields
            min_scale = click.prompt("Minimum number of replicas", default=1, type=int)
            max_scale = click.prompt("Maximum number of replicas", default=1, type=int)
            concurrency = click.prompt("Max concurrency (or leave blank)", default="", show_default=False)
            concurrency = int(concurrency) if concurrency else None

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
                env_vars=env_vars if env_vars else None,
                command=command,
            )

            created = cclient.create_inference(req)
            click.echo(f"Inference deployment {name} created with ID: {created.id}")

        elif depl_type == DeploymentType.COMPUTE_V2:
            # Select cluster using a numbered list
            clusters = cclient.get_clusters().results
            if not clusters:
                click.echo("No clusters available. Please ensure you have a cluster setup.")
                return

            click.echo("Available clusters:")
            for idx, cluster in enumerate(clusters, start=1):
                click.echo(f"{idx}. {cluster.display_name}")
            cluster_choice = click.prompt("Select a cluster by number", type=int, default=1)
            selected_cluster = clusters[cluster_choice - 1]
            cluster_id = selected_cluster.id

            # Hardware selection using a numbered list
            hw_resp = cclient.get_hardware_instances(cluster_id)
            if not hw_resp:
                click.echo("No hardware instances available for this cluster.")
                return

            click.echo("Available hardware instances:")
            for idx, hw in enumerate(hw_resp, start=1):
                click.echo(f"{idx}. {hw.name}")
            hw_choice = click.prompt("Select a hardware instance by number", type=int, default=1)
            selected_hw = hw_resp[hw_choice - 1]
            hw_id = selected_hw.id

            # Retrieve prebuilt images for compute deployments
            prebuilt_images = cclient.get_prebuilt_images(depl_type=depl_type)
            # Build list of image labels
            image_choices = [img.label for img in prebuilt_images.results] if prebuilt_images.results else []

            chosen_label = click.prompt(
                "Select a prebuilt image label",
                type=click.Choice(image_choices),
                show_choices=True,
                default=image_choices[0],
            )

            selected_prebuilt = next(img for img in prebuilt_images.results if img.label == chosen_label)

            # Find the prebuilt image with the matching label
            selected_prebuilt = next(img for img in prebuilt_images.results if img.label == chosen_label)
            # Prompt the user to select a tag from the available tags
            tag = click.prompt(
                "Select a tag for the image",
                type=click.Choice(selected_prebuilt.tags),
                show_choices=True,
                default=selected_prebuilt.tags[0],
            )
            # Combine the image URL with the chosen tag
            image_url = f"{selected_prebuilt.image_name}:{tag}"

            # For compute deployments, we might ask for a public SSH key
            ssh_key = click.prompt("Enter your public SSH key")

            # Right now we not support this on prod platform, just unify the feature
            #jupyter = click.prompt("Enable Jupyter Notebook on this compute deployment?", type=bool,default=False, show_default=False)

            from platform_api_python_client import CreateComputeDeploymentRequest

            req = CreateComputeDeploymentRequest(
                name=name,
                cluster_id=cluster_id,
                hardware_instance_id=hw_id,
                image_url=image_url,
                ssh_public_key=ssh_key,  # we require this
                #enable_jupyter=jupyter,
                )

            created = cclient.create_compute(req)
            click.echo(f"Compute deployment {name} created with ID: {created.id}")

        elif depl_type == DeploymentType.CSERVE:
            # Keep things simple, only use recipe.
            # Retrieve the recipe and hardware instances
            recipe = cclient.get_cserve_recipe()
            models = [r.model for r in recipe] if recipe else []

            if not models:
                click.echo("No models found in the recipe.")
                sys.exit(1)

            # --- Model Selection (Indexed) ---
            click.echo("Select a model:")
            for idx, m in enumerate(models, start=1):
                click.echo(f"{idx}. {m}")
            model_index = click.prompt("Enter the model number", type=int, default=1)
            if model_index < 1 or model_index > len(models):
                click.echo("Invalid model selection.")
                sys.exit(1)
            selected_model = models[model_index - 1]

            # --- Performance Option Selection (Indexed) ---
            perf_options = ["fastest", "cheapest", "best_value"]
            click.echo("Select performance option:")
            for idx, option in enumerate(perf_options, start=1):
                click.echo(f"{idx}. {option}")
            perf_index = click.prompt("Enter the performance option number", type=int, default=1)
            if perf_index < 1 or perf_index > len(perf_options):
                click.echo("Invalid performance selection.")
                sys.exit(1)
            selected_perf_option = perf_options[perf_index - 1]

            # Retrieve the recipe response for the selected model
            selected_response = next((r for r in recipe if r.model == selected_model), None)
            if not selected_response:
                click.echo("Selected model not found in recipe.")
                sys.exit(1)

            # Get the performance-specific recipe (this is a CServeRecipePerf instance)
            selected_perf = getattr(selected_response, selected_perf_option)

            # Retrieve the hardware instance ID from the selected performance option
            hardware_instance_id = selected_perf.hardware_instance_id

            # Get hardware instance details using cclient.get_hardware_instances()
            hw_instances = cclient.get_hardware_instances()
            selected_hw = next((hw for hw in hw_instances["results"] if hw["id"] == hardware_instance_id), None)
            if not selected_hw:
                click.echo(f"Hardware instance with id {hardware_instance_id} not found.")
                sys.exit(1)

            # Display the hardware instance information to the user
            click.echo("Selected Hardware Instance:")
            for key, value in selected_hw.items():
                click.echo(f"{key}: {value}")

            # Use the cluster_id from the hardware instance (no need to prompt the user)
            cluster_id = selected_hw["cluster_id"]

            # --- Additional Prompts ---
            # Prompt for Hugging Face token (if required)
            hf_token = click.prompt(
                "Enter your Hugging Face token or leave blank (if your model isn't private)",
                default="",
                show_default=False,
            )

            # Prompt for environment variables
            env_vars_str = click.prompt(
                "Enter environment variables in KEY=VALUE format (comma separated) or leave blank",
                default="",
                show_default=False,
            )
            env_vars = {}
            if env_vars_str.strip():
                for kv in env_vars_str.split(","):
                    try:
                        k, v = kv.strip().split("=")
                        env_vars[k] = v
                    except ValueError:
                        click.echo(f"Skipping invalid env var: {kv}")

            # Prompt for scaling and concurrency settings
            min_scale = click.prompt("Minimum number of replicas", default=1, type=int)
            max_scale = click.prompt("Maximum number of replicas", default=1, type=int)
            concurrency_input = click.prompt("Max concurrency (or leave blank)", default="", show_default=False)
            concurrency = int(concurrency_input) if concurrency_input else None

            # --- Create the Deployment Request ---
            from platform_api_python_client import CreateCServeDeploymentRequest

            req = CreateCServeDeploymentRequest(
                name=name,
                cluster_id=cluster_id,
                hardware_instance_id=hardware_instance_id,
                recipe=selected_perf.recipe,  # The underlying CServeV2Recipe instance
                hf_token=hf_token if hf_token.strip() else None,
                min_scale=min_scale,
                max_scale=max_scale,
                concurrency=concurrency,
                env_vars=env_vars if env_vars else None,
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
