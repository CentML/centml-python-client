from contextlib import contextmanager
from typing import Dict, Any, Optional

try:
    import platform_api_ops_client
    from platform_api_ops_client import OPSApi

    OPS_CLIENT_AVAILABLE = True
except ImportError:
    OPS_CLIENT_AVAILABLE = False

import platform_api_python_client

from centml.sdk import auth
from centml.sdk.config import settings


class CentMLOpsClient:
    """
    Client for CentML OPS API operations.
    Used for administrative tasks like managing CServe recipes.
    """

    def __init__(
        self,
        ops_api: Optional["OPSApi"] = None,
        external_api: Optional[platform_api_python_client.EXTERNALApi] = None,
    ):
        self._ops_api = ops_api
        self._external_api = external_api

    def get_clusters(self, include_hardware_instances: bool = False):
        """
        Get available clusters for the organization.

        Args:
            include_hardware_instances: If True, also fetch hardware instances for each cluster

        Returns:
            If include_hardware_instances=False: List of cluster configurations
            If include_hardware_instances=True: List of dicts with 'cluster' and 'hardware_instances' keys

        Example:
            with get_centml_ops_client() as ops_client:
                # Get clusters only
                clusters = ops_client.get_clusters()

                # Get clusters with hardware instances
                clusters_with_hw = ops_client.get_clusters(include_hardware_instances=True)
                for item in clusters_with_hw:
                    cluster = item['cluster']
                    print(f"Cluster {cluster.id}: {cluster.display_name}")
                    for hw in item['hardware_instances']:
                        print(f"  - {hw.name}: {hw.num_accelerators}x{hw.gpu_type}")
        """
        if self._external_api is None:
            raise RuntimeError("External API client not available")

        clusters = self._external_api.get_clusters_clusters_get().results

        if include_hardware_instances:
            result = []
            for cluster in clusters:
                hw_instances = (
                    self._external_api.get_hardware_instances_hardware_instances_get(
                        cluster_id=cluster.id
                    ).results
                )
                result.append({"cluster": cluster, "hardware_instances": hw_instances})
            return result

        return clusters

    def get_hardware_instances(self, cluster_id: Optional[int] = None):
        """
        Get hardware instances, optionally filtered by cluster.

        Args:
            cluster_id: Optional cluster ID to filter hardware instances

        Returns:
            List of hardware instance configurations

        Example:
            with get_centml_ops_client() as ops_client:
                # Get all hardware instances
                all_hw = ops_client.get_hardware_instances()

                # Get hardware instances for specific cluster
                cluster_hw = ops_client.get_hardware_instances(cluster_id=1000)
        """
        if self._external_api is None:
            raise RuntimeError("External API client not available")

        return self._external_api.get_hardware_instances_hardware_instances_get(
            cluster_id=cluster_id
        ).results

    def get_cserve_recipes(
        self, model: Optional[str] = None, hf_token: Optional[str] = None
    ):
        """
        Get CServe recipe configurations.

        Args:
            model: Optional model name to filter recipes (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            hf_token: Optional HuggingFace token for private models

        Returns:
            List of CServe recipe configurations

        Example:
            with get_centml_ops_client() as ops_client:
                # Get all recipes
                all_recipes = ops_client.get_cserve_recipes()

                # Get recipes for a specific model
                recipes = ops_client.get_cserve_recipes(model="meta-llama/Llama-3.3-70B-Instruct")
        """
        if self._external_api is None:
            raise RuntimeError("External API client not available")

        return self._external_api.get_cserve_recipe_deployments_cserve_recipes_get(
            model=model, hf_token=hf_token
        ).results

    def update_cserve_recipes(
        self, cluster_id: int, platform_data: Dict[str, Dict[str, Dict[str, Any]]]
    ):
        """
        Update CServe recipes from platform_db.json performance data.

        Args:
            cluster_id: The cluster ID to associate with hardware instances
            platform_data: Platform DB data in the format:
                {
                    "model_name": {
                        "fastest": {...},
                        "cheapest": {...},  # optional
                        "best_value": {...}  # optional
                    }
                }

        Returns:
            Response containing processed models and any errors

        Example:
            with get_centml_ops_client() as ops_client:
                with open('platform_db.json') as f:
                    platform_data = json.load(f)
                response = ops_client.update_cserve_recipes(cluster_id=1001, platform_data=platform_data)
                print(f"Processed: {response.processed_models}")
        """
        if self._ops_api is None:
            raise RuntimeError(
                "OPS API client not available. Install platform-api-ops-client."
            )

        return self._ops_api.update_cserve_recipes_ops_cserve_recipes_post(
            cluster_id=cluster_id, request_body=platform_data
        )

    def delete_cserve_recipe(self, model: str):
        """
        Delete CServe recipe configurations for a specific model.

        Args:
            model: The model name to delete (e.g., "meta-llama/Llama-3.3-70B-Instruct")

        Returns:
            Success response (200 OK)

        Example:
            with get_centml_ops_client() as ops_client:
                ops_client.delete_cserve_recipe(model="meta-llama/Llama-3.3-70B-Instruct")
        """
        if self._ops_api is None:
            raise RuntimeError(
                "OPS API client not available. Install platform-api-ops-client."
            )

        return self._ops_api.delete_cserve_recipe_ops_cserve_recipes_delete(model=model)


@contextmanager
def get_centml_ops_client():
    """
    Context manager for CentML OPS API client.

    This client provides:
    - get_clusters(): Get available clusters (uses external API, always available)
    - get_cserve_recipes(): Read recipes (uses external API, always available)
    - update_cserve_recipes(): Update recipes (requires platform-api-ops-client)
    - delete_cserve_recipe(): Delete recipes (requires platform-api-ops-client)

    Usage:
        with get_centml_ops_client() as ops_client:
            # Get clusters (always works)
            clusters = ops_client.get_clusters()

            # Get recipes (always works)
            recipes = ops_client.get_cserve_recipes(model="meta-llama/Llama-3.3-70B-Instruct")

            # Update/delete requires platform-api-ops-client
            response = ops_client.update_cserve_recipes(cluster_id=1001, platform_data=data)
    """
    configuration = platform_api_python_client.Configuration(
        host=settings.CENTML_PLATFORM_API_URL, access_token=auth.get_centml_token()
    )

    # Always initialize external API for read operations
    with platform_api_python_client.ApiClient(configuration) as external_client:
        external_api = platform_api_python_client.EXTERNALApi(external_client)

        # Initialize OPS API if available for write operations
        ops_api = None
        if OPS_CLIENT_AVAILABLE:
            ops_configuration = platform_api_ops_client.Configuration(
                host=settings.CENTML_PLATFORM_API_URL,
                access_token=auth.get_centml_token(),
            )
            with platform_api_ops_client.ApiClient(ops_configuration) as ops_client:
                ops_api = OPSApi(ops_client)
                yield CentMLOpsClient(ops_api=ops_api, external_api=external_api)
        else:
            # Still provide read-only functionality even without ops client
            yield CentMLOpsClient(ops_api=None, external_api=external_api)
