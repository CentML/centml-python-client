from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import platform_api_python_client
from platform_api_python_client import (
    CreateDynamoDeploymentRequest,
    CreateHardwareInstanceRequest,
    CreateJobDeploymentRequest,
    DeploymentType,
    UpdateClusterMetadataRequest,
)

from centml.sdk import ApiException
from centml.sdk.api import CentMLClient, get_centml_client
from centml.sdk.config import settings


def test_get_status_uses_v3_endpoint():
    api = MagicMock()
    expected_status = SimpleNamespace()
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.return_value = expected_status

    assert CentMLClient(api).get_status(123) is expected_status

    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_deployments_status_deployment_id_get.assert_not_called()


def test_get_status_falls_back_to_legacy_endpoint_when_v3_is_not_found():
    api = MagicMock()
    expected_status = SimpleNamespace()
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.side_effect = ApiException(status=404)
    api.get_deployment_status_deployments_status_deployment_id_get.return_value = expected_status

    assert CentMLClient(api).get_status(123) is expected_status

    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_deployments_status_deployment_id_get.assert_called_once_with(123)


def test_get_status_raises_v3_error_when_both_status_endpoints_fail():
    api = MagicMock()
    v3_error = ApiException(status=404)
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.side_effect = v3_error
    api.get_deployment_status_deployments_status_deployment_id_get.side_effect = ApiException(status=404)

    try:
        CentMLClient(api).get_status(123)
    except ApiException as e:
        assert e is v3_error
    else:
        raise AssertionError("Expected ApiException")

    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_deployments_status_deployment_id_get.assert_called_once_with(123)


def test_get_job_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.get_job_deployment_deployments_job_deployment_id_get.return_value = expected_response
    client = CentMLClient(api)

    response = client.get_job(123)

    assert response is expected_response
    api.get_job_deployment_deployments_job_deployment_id_get.assert_called_once_with(123)


def test_create_job_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.create_job_deployment_deployments_job_post.return_value = expected_response
    request = CreateJobDeploymentRequest(
        name="test-job", cluster_id=1, hardware_instance_id=2, image_url="registry.example.com/job:latest"
    )
    client = CentMLClient(api)

    response = client.create_job(request)

    assert response is expected_response
    api.create_job_deployment_deployments_job_post.assert_called_once_with(request)


def _dynamo_request():
    return CreateDynamoDeploymentRequest(
        name="test-dynamo",
        cluster_id=1,
        hardware_instance_id=2,
        model="Qwen/Qwen3-0.6B",
        endpoint_bearer_token="test-only",
    )


def test_generated_client_exposes_dynamo_contract():
    assert DeploymentType.DYNAMO.value == "dynamo"
    assert hasattr(platform_api_python_client.EXTERNALApi, "get_dynamo_deployment_deployments_dynamo_deployment_id_get")
    assert hasattr(platform_api_python_client.EXTERNALApi, "create_dynamo_deployment_deployments_dynamo_post")
    assert hasattr(platform_api_python_client.EXTERNALApi, "update_dynamo_deployment_deployments_dynamo_put")


def test_get_dynamo_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.get_dynamo_deployment_deployments_dynamo_deployment_id_get.return_value = expected_response
    client = CentMLClient(api)

    response = client.get_dynamo(123)

    assert response is expected_response
    api.get_dynamo_deployment_deployments_dynamo_deployment_id_get.assert_called_once_with(123)


def test_create_dynamo_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.create_dynamo_deployment_deployments_dynamo_post.return_value = expected_response
    request = _dynamo_request()
    client = CentMLClient(api)

    response = client.create_dynamo(request)

    assert response is expected_response
    api.create_dynamo_deployment_deployments_dynamo_post.assert_called_once_with(request)


def test_update_dynamo_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.update_dynamo_deployment_deployments_dynamo_put.return_value = expected_response
    request = _dynamo_request()
    client = CentMLClient(api)

    response = client.update_dynamo(123, request)

    assert response is expected_response
    api.update_dynamo_deployment_deployments_dynamo_put.assert_called_once_with(123, request)


def test_get_centml_client_uses_authenticated_generated_client():
    configuration = MagicMock()
    api_client_context = MagicMock()
    generated_api_client = MagicMock()
    generated_external_api = MagicMock()
    expected_clusters = MagicMock()
    generated_external_api.get_clusters_clusters_get.return_value = expected_clusters
    api_client_context.__enter__.return_value = generated_api_client

    with (
        patch("centml.sdk.api.auth.get_centml_token", return_value="test-access-token") as get_token,
        patch(
            "centml.sdk.api.platform_api_python_client.Configuration", return_value=configuration
        ) as configuration_cls,
        patch("centml.sdk.api.platform_api_python_client.ApiClient", return_value=api_client_context) as api_client_cls,
        patch(
            "centml.sdk.api.platform_api_python_client.EXTERNALApi", return_value=generated_external_api
        ) as external_api_cls,
    ):
        with get_centml_client() as client:
            assert client.get_clusters() is expected_clusters

    get_token.assert_called_once_with()
    configuration_cls.assert_called_once_with(host=settings.CENTML_PLATFORM_API_URL, access_token="test-access-token")
    api_client_cls.assert_called_once_with(configuration)
    external_api_cls.assert_called_once_with(generated_api_client)


def test_generated_client_exposes_cluster_metadata_contract():
    assert hasattr(
        platform_api_python_client.EXTERNALApi, "update_cluster_metadata_clusters_cluster_id_metadata_put"
    )
    assert hasattr(platform_api_python_client, "UpdateClusterMetadataRequest")


def test_update_cluster_metadata_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.update_cluster_metadata_clusters_cluster_id_metadata_put.return_value = expected_response
    request = UpdateClusterMetadataRequest(deployment_creation_disabled=True)
    client = CentMLClient(api)

    response = client.update_cluster_metadata(42, request)

    assert response is expected_response
    api.update_cluster_metadata_clusters_cluster_id_metadata_put.assert_called_once_with(42, request)


def test_get_hardware_instances_returns_results():
    api = MagicMock()
    expected_results = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    api.get_hardware_instances_hardware_instances_get.return_value = SimpleNamespace(results=expected_results)
    client = CentMLClient(api)

    response = client.get_hardware_instances(cluster_id=5)

    assert response is expected_results
    api.get_hardware_instances_hardware_instances_get.assert_called_once_with(cluster_id=5)


def test_create_hardware_instance_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.create_hardware_instance_hardware_instances_post.return_value = expected_response
    request = CreateHardwareInstanceRequest(
        cluster_id=1,
        name="h100-test",
        gpu_type="H100",
        num_gpu=8,
        cpu=64000,
        memory=128000,
        accelerator_resource_key="nvidia.com/gpu",
        node_affinity_labels={"gpu": "h100"},
        accelerator_memory=80000,
    )
    client = CentMLClient(api)

    response = client.create_hardware_instance(request)

    assert response is expected_response
    api.create_hardware_instance_hardware_instances_post.assert_called_once_with(request)


def test_delete_hardware_instance_delegates_to_platform_client():
    api = MagicMock()
    expected_response = MagicMock()
    api.delete_hardware_instance_hardware_instances_hardware_instance_id_delete.return_value = expected_response
    client = CentMLClient(api)

    response = client.delete_hardware_instance(123)

    assert response is expected_response
    api.delete_hardware_instance_hardware_instances_hardware_instance_id_delete.assert_called_once_with(123)
