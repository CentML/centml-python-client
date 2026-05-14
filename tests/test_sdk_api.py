from types import SimpleNamespace
from unittest.mock import MagicMock

from centml.sdk import DeploymentType
from centml.sdk.api import CentMLClient


def test_get_status_uses_v3_endpoint_when_deployment_type_is_v3():
    api = MagicMock()
    deployment = SimpleNamespace(id=123, type=DeploymentType.INFERENCE_V3)
    expected_status = SimpleNamespace()
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.return_value = expected_status

    assert CentMLClient(api).get_status(deployment) is expected_status

    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_deployments_status_deployment_id_get.assert_not_called()


def test_get_status_uses_legacy_endpoint_when_deployment_type_is_legacy():
    api = MagicMock()
    deployment = SimpleNamespace(id=123, type=DeploymentType.INFERENCE)
    expected_status = SimpleNamespace(type=DeploymentType.INFERENCE)
    api.get_deployment_status_deployments_status_deployment_id_get.return_value = expected_status

    assert CentMLClient(api).get_status(deployment) is expected_status

    api.get_deployment_status_deployments_status_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_not_called()


def test_get_status_discovers_type_for_id_only_callers():
    api = MagicMock()
    legacy_status = SimpleNamespace(type=DeploymentType.CSERVE_V3)
    expected_status = SimpleNamespace()
    api.get_deployment_status_deployments_status_deployment_id_get.return_value = legacy_status
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.return_value = expected_status

    assert CentMLClient(api).get_status(123) is expected_status

    api.get_deployment_status_deployments_status_deployment_id_get.assert_called_once_with(123)
    api.get_deployment_status_v3_deployments_status_v3_deployment_id_get.assert_called_once_with(123)
