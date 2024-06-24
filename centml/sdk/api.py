import contextlib
import platform_api_client
from platform_api_client.models.deployment_status import DeploymentStatus

from . import auth
from .config import Config
from .utils import client_certs


@contextlib.contextmanager
def get_api():
    configuration = platform_api_client.Configuration(host=Config.platformapi_url, access_token=auth.get_centml_token())

    with platform_api_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_client.EXTERNALApi(api_client)

        yield api_instance


def get(depl_type):
    with get_api() as api:
        results = api.get_deployments_deployments_get(type=depl_type).results
        deployments = sorted(results, reverse=True, key=lambda d: d.created_at)

        rows = [
            [d.id, d.name, d.type.value, d.status.value, d.created_at.strftime("%Y-%m-%d %H:%M:%S")]
            for d in deployments
        ]

        return rows


def get_status(id):
    with get_api() as api:
        return api.get_deployment_status_deployments_status_deployment_id_get(id)


def get_inference(id):
    with get_api() as api:
        return api.get_inference_deployment_deployments_inference_deployment_id_get(id)


def get_compute(id):
    with get_api() as api:
        return api.get_compute_deployment_deployments_compute_deployment_id_get(id)


def create_inference(
    name, image, port, is_private, hw_to_id_map, health, min_replicas, max_replicas, env, command, command_args, timeout
):  
    triplet = None
    if is_private:
        triplet = client_certs.generate_ca_client_triplet(name)
        # Handle automatic download of client private secrets
        client_certs.save_pem_file(name, triplet.client_private_key, triplet.client_certificate)
    with get_api() as api:
        req = platform_api_client.CreateInferenceDeploymentRequest(
            name=name,
            image_url=image,
            port=port,
            hardware_instance_id=hw_to_id_map,
            healthcheck=health,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            env_vars=dict(env) if dict(env) else None,
            command=[command] if command else None,
            command_args=(list(command_args) if command and len(list(command_args)) > 0 else None),
            timeout=timeout,
            endpoint_certificate_authority=triplet.certificate_authority if triplet else None,
        )
        return api.create_inference_deployment_deployments_inference_post(req)


def create_compute(name, image, username, password, ssh_key, hw_to_id_map):
    with get_api() as api:
        req = platform_api_client.CreateComputeDeploymentRequest(
            name=name,
            image_url=image,
            hardware_instance_id=hw_to_id_map,
            username=username,
            password=password,
            ssh_key=ssh_key if ssh_key else None,
        )
        return api.create_compute_deployment_deployments_compute_post(req)


def update_status(id, new_status):
    with get_api() as api:
        status_req = platform_api_client.DeploymentStatusRequest(status=new_status)
        api.update_deployment_status_deployments_status_deployment_id_put(id, status_req)


def delete(id):
    update_status(id, DeploymentStatus.DELETED)


def pause(id):
    update_status(id, DeploymentStatus.PAUSED)


def resume(id):
    update_status(id, DeploymentStatus.ACTIVE)
