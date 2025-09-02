from contextlib import contextmanager
from typing import Union

import platform_api_python_client
from platform_api_python_client import (
    DeploymentType,
    DeploymentStatus,
    CreateInferenceDeploymentRequest,
    CreateInferenceV3DeploymentRequest,
    CreateComputeDeploymentRequest,
    CreateCServeV2DeploymentRequest,
    CreateCServeV3DeploymentRequest,
    ApiException,
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
        """Get Inference deployment details - automatically handles both V2 and V3 deployments"""
        # Try V3 first (recommended), fallback to V2 if deployment is V2
        try:
            return self._api.get_inference_v3_deployment_deployments_inference_v3_deployment_id_get(id)
        except ApiException as e:
            # If V3 fails with 404 or similar, try V2
            if e.status in [404, 400]:  # Deployment might be V2 or endpoint not found
                try:
                    return self._api.get_inference_deployment_deployments_inference_deployment_id_get(id)
                except ApiException as v2_error:
                    # If both fail, raise the original V3 error as it's more likely to be the real issue
                    raise e from v2_error
            else:
                # For other errors (auth, network, etc.), raise immediately
                raise

    def get_compute(self, id):
        return self._api.get_compute_deployment_deployments_compute_deployment_id_get(id)

    def get_cserve(self, id):
        """Get CServe deployment details - automatically handles both V2 and V3 deployments"""
        # Try V3 first (recommended), fallback to V2 if deployment is V2
        try:
            return self._api.get_cserve_v3_deployment_deployments_cserve_v3_deployment_id_get(id)
        except ApiException as e:
            # If V3 fails with 404 or similar, try V2
            if e.status in [404, 400]:  # Deployment might be V2 or endpoint not found
                try:
                    return self._api.get_cserve_v2_deployment_deployments_cserve_v2_deployment_id_get(id)
                except ApiException as v2_error:
                    # If both fail, raise the original V3 error as it's more likely to be the real issue
                    raise e from v2_error
            else:
                # For other errors (auth, network, etc.), raise immediately
                raise

    def create_inference(self, request: CreateInferenceV3DeploymentRequest):
        return self._api.create_inference_v3_deployment_deployments_inference_v3_post(request)

    def create_inference_v2(self, request: CreateInferenceDeploymentRequest):
        return self._api.create_inference_deployment_deployments_inference_post(request)

    def create_inference_v3(self, request: CreateInferenceV3DeploymentRequest):
        return self._api.create_inference_v3_deployment_deployments_inference_v3_post(request)

    def create_compute(self, request: CreateComputeDeploymentRequest):
        return self._api.create_compute_deployment_deployments_compute_post(request)

    def create_cserve(self, request: CreateCServeV3DeploymentRequest):
        return self._api.create_cserve_v3_deployment_deployments_cserve_v3_post(request)

    def create_cserve_v2(self, request: CreateCServeV2DeploymentRequest):
        return self._api.create_cserve_v2_deployment_deployments_cserve_v2_post(request)

    def create_cserve_v3(self, request: CreateCServeV3DeploymentRequest):
        return self._api.create_cserve_v3_deployment_deployments_cserve_v3_post(request)

    def detect_inference_deployment_version(self, deployment_id: int) -> str:
        """Detect if an inference deployment is V2 or V3 by testing the specific API endpoints"""
        try:
            # Try V3 endpoint first
            self._api.get_inference_v3_deployment_deployments_inference_v3_deployment_id_get(deployment_id)
            return 'v3'
        except ApiException as e:
            if e.status in [404, 400]:  # V3 endpoint doesn't exist for this deployment
                try:
                    # Try V2 endpoint
                    self._api.get_inference_deployment_deployments_inference_deployment_id_get(deployment_id)
                    return 'v2'
                except ApiException as exc:
                    # If both fail, it might not be an inference deployment or doesn't exist
                    raise ValueError(
                        f"Deployment {deployment_id} is not a valid inference deployment or does not exist"
                    ) from exc
            else:
                # Other error (auth, network, etc.)
                raise

    def update_inference(
        self, deployment_id: int, request: Union[CreateInferenceDeploymentRequest, CreateInferenceV3DeploymentRequest]
    ):
        """Update Inference deployment - validates request type matches deployment version"""
        # Detect the deployment version
        deployment_version = self.detect_inference_deployment_version(deployment_id)

        # Validate request type matches deployment version
        if isinstance(request, CreateInferenceV3DeploymentRequest):
            if deployment_version != 'v3':
                raise ValueError(
                    f"Deployment {deployment_id} is Inference {deployment_version.upper()}, "
                    f"but you provided a V3 request. Please use CreateInferenceDeploymentRequest instead."
                )
            return self._api.update_inference_v3_deployment_deployments_inference_v3_put(
                deployment_id, request
            )
        elif isinstance(request, CreateInferenceDeploymentRequest):
            if deployment_version != 'v2':
                raise ValueError(
                    f"Deployment {deployment_id} is Inference {deployment_version.upper()}, "
                    f"but you provided a V2 request. Please use CreateInferenceV3DeploymentRequest instead."
                )
            return self._api.update_inference_deployment_deployments_inference_put(
                deployment_id, request
            )
        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}. "
                f"Expected CreateInferenceDeploymentRequest or CreateInferenceV3DeploymentRequest."
            )

    def update_compute(self, deployment_id: int, request: CreateComputeDeploymentRequest):
        return self._api.update_compute_deployment_deployments_compute_put(deployment_id, request)

    def detect_deployment_version(self, deployment_id: int) -> str:
        """Detect if a deployment is V2 or V3 by testing the specific API endpoints"""
        try:
            # Try V3 endpoint first
            self._api.get_cserve_v3_deployment_deployments_cserve_v3_deployment_id_get(deployment_id)
            return 'v3'
        except ApiException as e:
            if e.status in [404, 400]:  # V3 endpoint doesn't exist for this deployment
                try:
                    # Try V2 endpoint
                    self._api.get_cserve_v2_deployment_deployments_cserve_v2_deployment_id_get(deployment_id)
                    return 'v2'
                except ApiException as exc:
                    # If both fail, it might not be a CServe deployment or doesn't exist
                    raise ValueError(
                        f"Deployment {deployment_id} is not a valid CServe deployment or does not exist"
                    ) from exc
            else:
                # Other error (auth, network, etc.)
                raise

    def update_cserve(
        self, deployment_id: int, request: Union[CreateCServeV2DeploymentRequest, CreateCServeV3DeploymentRequest]
    ):
        """Update CServe deployment - validates request type matches deployment version"""
        # Detect the deployment version
        deployment_version = self.detect_deployment_version(deployment_id)

        # Validate request type matches deployment version
        if isinstance(request, CreateCServeV3DeploymentRequest):
            if deployment_version != 'v3':
                raise ValueError(
                    f"Deployment {deployment_id} is CServe {deployment_version.upper()}, "
                    f"but you provided a V3 request. Please use CreateCServeV2DeploymentRequest instead."
                )
            return self._api.update_cserve_v3_deployment_deployments_cserve_v3_put(
                deployment_id, request
            )
        elif isinstance(request, CreateCServeV2DeploymentRequest):
            if deployment_version != 'v2':
                raise ValueError(
                    f"Deployment {deployment_id} is CServe {deployment_version.upper()}, "
                    f"but you provided a V2 request. Please use CreateCServeV3DeploymentRequest instead."
                )
            return self._api.update_cserve_v2_deployment_deployments_cserve_v2_put(
                deployment_id, request
            )
        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}. "
                f"Expected CreateCServeV2DeploymentRequest or CreateCServeV3DeploymentRequest."
            )

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

    def detect_cserve_deployment_version(self, deployment_response):
        """Detect if a CServe deployment is V2 or V3 based on response fields"""
        # Check for V3-specific fields
        if hasattr(deployment_response, 'max_surge') or hasattr(deployment_response, 'max_unavailable'):
            return 'v3'
        # Check for V3 field names (min_replicas vs min_scale)
        if hasattr(deployment_response, 'min_replicas'):
            return 'v3'
        # Check for V2 field names
        if hasattr(deployment_response, 'min_scale'):
            return 'v2'
        # Default to V2 for backward compatibility
        return 'v2'

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
