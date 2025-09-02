from contextlib import contextmanager
from typing import Union

import platform_api_python_client
from platform_api_python_client import (
    DeploymentType,
    DeploymentStatus,
    CreateInferenceDeploymentRequest,
    CreateComputeDeploymentRequest,
    CreateCServeV2DeploymentRequest,
    CreateCServeV3DeploymentRequest,
    CServeV2Recipe,
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
        return self._api.get_inference_deployment_deployments_inference_deployment_id_get(id)

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
                    raise e
            else:
                # For other errors (auth, network, etc.), raise immediately
                raise

    def create_inference(self, request: CreateInferenceDeploymentRequest):
        return self._api.create_inference_deployment_deployments_inference_post(request)

    def create_compute(self, request: CreateComputeDeploymentRequest):
        return self._api.create_compute_deployment_deployments_compute_post(request)

    def create_cserve(self, request: CreateCServeV3DeploymentRequest):
        return self._api.create_cserve_v3_deployment_deployments_cserve_v3_post(request)

    def create_cserve_v2(self, request: CreateCServeV2DeploymentRequest):
        return self._api.create_cserve_v2_deployment_deployments_cserve_v2_post(request)

    def create_cserve_v3(self, request: CreateCServeV3DeploymentRequest):
        return self._api.create_cserve_v3_deployment_deployments_cserve_v3_post(request)

    def update_inference(self, deployment_id: int, request: CreateInferenceDeploymentRequest):
        return self._api.update_inference_deployment_deployments_inference_put(deployment_id, request)

    def update_compute(self, deployment_id: int, request: CreateComputeDeploymentRequest):
        return self._api.update_compute_deployment_deployments_compute_put(deployment_id, request)

    def update_cserve(
        self, deployment_id: int, request: Union[CreateCServeV2DeploymentRequest, CreateCServeV3DeploymentRequest]
    ):
        """Update CServe deployment - automatically handles both V2 and V3 deployments"""
        # Determine the approach based on the request type
        if isinstance(request, CreateCServeV3DeploymentRequest):
            # V3 request - try V3 API first, fallback if deployment is actually V2
            try:
                return self._api.update_cserve_v3_deployment_deployments_cserve_v3_put(deployment_id, request)
            except ApiException as e:
                if e.status in [404, 400]:  # V3 API failed, deployment might be V2
                    # Convert V3 request to V2 and try V2 API
                    v2_request = self._convert_v3_to_v2_request(request)
                    return self._api.update_cserve_v2_deployment_deployments_cserve_v2_put(deployment_id, v2_request)
                else:
                    raise
        elif isinstance(request, CreateCServeV2DeploymentRequest):
            # V2 request - try V2 API first, fallback to V3 if deployment is actually V3
            try:
                return self._api.update_cserve_v2_deployment_deployments_cserve_v2_put(deployment_id, request)
            except ApiException as e:
                if e.status in [404, 400]:  # V2 API failed, deployment might be V3
                    # Convert V2 request to V3 and try V3 API
                    v3_request = self.convert_v2_to_v3_request(request)
                    return self._api.update_cserve_v3_deployment_deployments_cserve_v3_put(deployment_id, v3_request)
                else:
                    raise
        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}. Expected CreateCServeV2DeploymentRequest or CreateCServeV3DeploymentRequest."
            )

    def _convert_v3_to_v2_request(self, v3_request: CreateCServeV3DeploymentRequest) -> CreateCServeV2DeploymentRequest:
        """Convert V3 request format to V2 format (reverse of convert_v2_to_v3_request)"""
        # Get all fields from V3 request
        kwargs = v3_request.model_dump() if hasattr(v3_request, 'model_dump') else v3_request.dict()

        # Remove old V3 field names
        min_replicas = kwargs.pop('min_replicas', None)
        max_replicas = kwargs.pop('max_replicas', None)
        initial_replicas = kwargs.pop('initial_replicas', None)
        # Remove V3-only fields
        kwargs.pop('max_surge', None)
        kwargs.pop('max_unavailable', None)

        # Add new V2 field names
        kwargs['min_scale'] = min_replicas
        kwargs['max_scale'] = max_replicas
        if initial_replicas is not None:
            kwargs['initial_scale'] = initial_replicas

        return CreateCServeV2DeploymentRequest(**kwargs)

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

    def convert_v2_to_v3_request(self, v2_request: CreateCServeV2DeploymentRequest) -> CreateCServeV3DeploymentRequest:
        """Convert V2 request format to V3 format with field mapping"""
        # Get all fields from V2 request
        kwargs = v2_request.model_dump() if hasattr(v2_request, 'model_dump') else v2_request.dict()

        # Remove old V2 field names
        min_scale = kwargs.pop('min_scale', None)
        max_scale = kwargs.pop('max_scale', None)
        initial_scale = kwargs.pop('initial_scale', None)

        # Add new V3 field names
        kwargs['min_replicas'] = min_scale
        kwargs['max_replicas'] = max_scale
        if initial_scale is not None:
            kwargs['initial_replicas'] = initial_scale

        # Add V3-specific fields
        kwargs['max_surge'] = None
        kwargs['max_unavailable'] = None

        return CreateCServeV3DeploymentRequest(**kwargs)

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
