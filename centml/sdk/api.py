from contextlib import contextmanager

import platform_api_python_client
from platform_api_python_client import (
    DeploymentStatus,
    CreateInferenceDeploymentV2Request,
    CreateComputeDeploymentV2Request,
    CreateCServeDeploymentRequest,
)

from centml.sdk import auth
from centml.sdk.config import settings


class CentMLClient:
    def __init__(self, api):
        self._api = api

    def get(self, depl_type):
        results = self._api.get_deployments_deployments_v2_get(type=depl_type).results
        deployments = sorted(results, reverse=True, key=lambda d: d.created_at)
        return deployments

    def get_status(self, id):
        return self._api.get_deployment_status_deployments_v2_status_deployment_id_get(id)

    def get_inference(self, id):
        return self._api.get_inference_deployment_deployments_v2_inference_deployment_id_get(id)

    def get_compute(self, id):
        return self._api.get_compute_deployment_deployments_v2_compute_deployment_id_get(id)

    def get_cserve(self, id):
        return self._api.get_cserve_deployment_deployments_v2_cserve_deployment_id_get(id)

    def create_inference(self, request: CreateInferenceDeploymentV2Request):
        return self._api.create_inference_deployment_deployments_v2_inference_post(request)

    def create_compute(self, request: CreateComputeDeploymentV2Request):
        return self._api.create_compute_deployment_deployments_compute_post(request)

    def create_cserve(self, request: CreateCServeDeploymentRequest):
        return self._api.create_cserve_deployment_deployments_v2_cserve_post(request)

    def _update_status(self, id, new_status):
        status_req = platform_api_python_client.DeploymentStatusRequest(status=new_status)
        self._api.update_deployment_status_deployments_v2_status_deployment_id_put(id, status_req)

    def delete(self, id):
        self._update_status(id, DeploymentStatus.DELETED)

    def pause(self, id):
        self._update_status(id, DeploymentStatus.PAUSED)

    def resume(self, id):
        self._update_status(id, DeploymentStatus.ACTIVE)

    def get_clusters(self):
        return self._api.get_clusters_clusters_get()

    def get_hardware_instances(self, cluster_id):
        return self._api.get_hardware_instances_hardware_instances_v2_get(cluster_id).results


@contextmanager
def get_centml_client():
    configuration = platform_api_python_client.Configuration(
        host=settings.CENTML_PLATFORM_API_URL, access_token=auth.get_centml_token()
    )

    with platform_api_python_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_python_client.EXTERNALApi(api_client)

        yield CentMLClient(api_instance)
