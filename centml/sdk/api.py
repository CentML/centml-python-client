import contextlib
import platform_api_client
from platform_api_client.models.deployment_status import DeploymentStatus

from . import auth
from .config import Config


@contextlib.contextmanager
def get_api():
    configuration = platform_api_client.Configuration(
        host=Config.platformapi_url, access_token=auth.get_centml_token()
    )

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
    name,
    image,
    port,
    hw_id,
    health,
    min_replicas,
    max_replicas,
    username,
    password,
    env):
    with get_api() as api:
        req = platform_api_client.CreateInferenceDeploymentRequest(
            name=name,
            image_url=image,
            hardware_instance_id=hw_id,
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
        return api.create_inference_deployment_deployments_inference_post(req)


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
