from types import SimpleNamespace
from unittest.mock import MagicMock

from platform_api_python_client import CreateJobDeploymentRequest

from centml.sdk import ApiException
from centml.sdk.api import CentMLClient


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
        name="test-job",
        cluster_id=1,
        hardware_instance_id=2,
        image_url="registry.example.com/job:latest",
    )
    client = CentMLClient(api)

    response = client.create_job(request)

    assert response is expected_response
    api.create_job_deployment_deployments_job_post.assert_called_once_with(request)
