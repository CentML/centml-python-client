from contextlib import contextmanager

import platform_api_python_client
from platform_api_python_client import (
    DeploymentType,
    DeploymentStatus,
    CreateInferenceDeploymentRequest,
    CreateComputeDeploymentRequest,
    CreateCServeV2DeploymentRequest,
    Metric,
)

from centml.sdk import auth
from centml.sdk.config import settings


class CentMLClient:
    def __init__(self, api):
        self._api: platform_api_python_client.EXTERNALApi = api

    def get(self, depl_type):
        results = self._api.get_deployments_deployments_get(type=depl_type).results
        deployments = sorted(results, reverse=True, key=lambda d: d.created_at)
        return deployments

    def get_status(self, id):
        return self._api.get_deployment_status_deployments_status_deployment_id_get(id)

    def get_inference(self, id):
        return self._api.get_inference_deployment_deployments_inference_deployment_id_get(id)

    def get_compute(self, id):
        return self._api.get_compute_deployment_deployments_compute_deployment_id_get(id)

    def get_cserve(self, id):
        return self._api.get_cserve_v2_deployment_deployments_cserve_v2_deployment_id_get(id)

    def create_inference(self, request: CreateInferenceDeploymentRequest):
        return self._api.create_inference_deployment_deployments_inference_post(request)

    def create_compute(self, request: CreateComputeDeploymentRequest):
        return self._api.create_compute_deployment_deployments_compute_post(request)

    def create_cserve(self, request: CreateCServeV2DeploymentRequest):
        return self._api.create_cserve_v2_deployment_deployments_cserve_v2_post(request)

    def update_inference(self, deployment_id: int, request: CreateInferenceDeploymentRequest):
        return self._api.update_inference_deployment_deployments_inference_put(deployment_id, request)

    def update_compute(self, deployment_id: int, request: CreateComputeDeploymentRequest):
        return self._api.update_compute_deployment_deployments_compute_put(deployment_id, request)

    def update_cserve(self, deployment_id: int, request: CreateCServeV2DeploymentRequest):
        return self._api.update_cserve_v2_deployment_deployments_cserve_v2_put(deployment_id, request)

    def _update_status(self, id, new_status):
        status_req = platform_api_python_client.DeploymentStatusRequest(status=new_status)
        self._api.update_deployment_status_deployments_status_deployment_id_put(id, status_req)

    def delete(self, id):
        self._update_status(id, DeploymentStatus.DELETED)

    def pause(self, id):
        self._update_status(id, DeploymentStatus.PAUSED)

    def resume(self, id):
        self._update_status(id, DeploymentStatus.ACTIVE)

    def get_clusters(self):
        return self._api.get_clusters_clusters_get()

    def get_hardware_instances(self, cluster_id=None):
        return self._api.get_hardware_instances_hardware_instances_get(
            cluster_id=cluster_id if cluster_id else None
        ).results

    def get_prebuilt_images(self, depl_type: DeploymentType):
        return self._api.get_prebuilt_images_prebuilt_images_get(type=depl_type)

    def get_cserve_recipe(self, model=None, hf_token=None):
        return self._api.get_cserve_recipe_deployments_cserve_recipes_get(model=model, hf_token=hf_token).results

    def get_cluster_id(self, hardware_instance_id):
        filtered_hw = list(filter(lambda h: h.id == hardware_instance_id, self.get_hardware_instances()))

        if len(filtered_hw) == 0:
            raise Exception(f"Invalid hardware instance id {hardware_instance_id}")

        return filtered_hw[0].cluster_id

    def get_user_vault(self, type):
        items = self._api.get_all_user_vault_items_endpoint_user_vault_get(type).results

        return {i.key: i.value for i in items}

    # pylint: disable=R0917
    def get_deployment_usage(
        self, id: int, metric: Metric, start_time_in_seconds: int, end_time_in_seconds: int, step: int
    ):
        return self._api.get_usage_deployments_usage_deployment_id_get(
            deployment_id=id,
            metric=metric,
            start_time_in_seconds=start_time_in_seconds,
            end_time_in_seconds=end_time_in_seconds,
            step=step,
        ).values


@contextmanager
def get_centml_client():
    configuration = platform_api_python_client.Configuration(
        host=settings.CENTML_PLATFORM_API_URL, access_token=auth.get_centml_token()
    )

    with platform_api_python_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_python_client.EXTERNALApi(api_client)

        yield CentMLClient(api_instance)
