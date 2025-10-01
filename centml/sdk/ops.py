from contextlib import contextmanager
from typing import Dict, Any

try:
    import platform_api_ops_client
    from platform_api_ops_client import OPSApi
    OPS_CLIENT_AVAILABLE = True
except ImportError:
    OPS_CLIENT_AVAILABLE = False

from centml.sdk import auth
from centml.sdk.config import settings


class CentMLOpsClient:
    """
    Client for CentML OPS API operations.
    Used for administrative tasks like managing CServe recipes.
    """

    def __init__(self, api: "OPSApi"):
        self._api = api

    def update_cserve_recipes(self, cluster_id: int, platform_data: Dict[str, Dict[str, Dict[str, Any]]]):
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
        return self._api.update_cserve_recipes_ops_cserve_recipes_post(
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
        return self._api.delete_cserve_recipe_ops_cserve_recipes_delete(model=model)


@contextmanager
def get_centml_ops_client():
    """
    Context manager for CentML OPS API client.
    Requires platform-api-ops-client to be installed.

    Usage:
        with get_centml_ops_client() as ops_client:
            response = ops_client.update_cserve_recipes(cluster_id=1001, platform_data=data)
    """
    if not OPS_CLIENT_AVAILABLE:
        raise ImportError(
            "platform-api-ops-client is required for OPS operations. "
            "Install it with: pip install platform-api-ops-client"
        )

    configuration = platform_api_ops_client.Configuration(
        host=settings.CENTML_PLATFORM_API_URL, access_token=auth.get_centml_token()
    )

    with platform_api_ops_client.ApiClient(configuration) as api_client:
        api_instance = OPSApi(api_client)

        yield CentMLOpsClient(api_instance)

