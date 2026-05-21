from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from centml.cli.cluster import _get_service_status, _get_status_error_messages
from centml.sdk import DeploymentStatus, DeploymentType, RolloutStatus, ServiceStatus


def test_service_status_uses_legacy_service_status_when_present():
    status_response = SimpleNamespace(service_status=ServiceStatus.HEALTHY)

    assert _get_service_status(status_response, revision_number=None) == ServiceStatus.HEALTHY


def test_service_status_maps_v3_healthy_rollout_status():
    status_response = SimpleNamespace(rollout_status=RolloutStatus.HEALTHY)

    assert _get_service_status(status_response, revision_number=None) == ServiceStatus.HEALTHY


def test_service_status_uses_current_v3_revision_status():
    status_response = SimpleNamespace(
        rollout_status=RolloutStatus.PROGRESSING,
        revision_pod_details_list=[
            SimpleNamespace(revision_number=1, revision_status=ServiceStatus.HEALTHY),
            SimpleNamespace(revision_number=2, revision_status=ServiceStatus.SCALINGUP),
        ],
    )

    assert _get_service_status(status_response, revision_number=2) == ServiceStatus.SCALINGUP


def test_service_status_uses_v3_revision_without_revision_number_as_fallback():
    status_response = SimpleNamespace(
        rollout_status=RolloutStatus.DEGRADED,
        revision_pod_details_list=[
            SimpleNamespace(revision_number=None, revision_status=ServiceStatus.IMAGEPULLBACKOFF)
        ],
    )

    assert _get_service_status(status_response, revision_number=2) == ServiceStatus.IMAGEPULLBACKOFF


def test_status_error_messages_include_revision_and_pod_messages():
    status_response = SimpleNamespace(
        revision_pod_details_list=[
            SimpleNamespace(
                revision_number=3,
                error_message="revision failed",
                pod_details_list=[
                    SimpleNamespace(name="pod-a", error_message="image pull failed"),
                    SimpleNamespace(name="pod-b", error_message=None),
                ],
            )
        ]
    )

    messages = _get_status_error_messages(status_response)

    assert messages == ["revision 3: revision failed", "revision 3 / pod-a: image pull failed"]


def test_status_error_messages_do_not_repeat_duplicate_messages():
    duplicate_message = "one or more objects failed to apply"
    status_response = SimpleNamespace(
        revision_pod_details_list=[
            SimpleNamespace(
                revision_number=None,
                error_message=duplicate_message,
                pod_details_list=[SimpleNamespace(name=None, error_message=duplicate_message)],
            )
        ]
    )

    assert _get_status_error_messages(status_response) == [f"revision: {duplicate_message}"]


def test_status_error_messages_include_legacy_status_message():
    status_response = SimpleNamespace(error_message="legacy service failure")

    assert _get_status_error_messages(status_response) == ["legacy service failure"]


@contextmanager
def _patch_cluster_client():
    client = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = client
    context.__exit__.return_value = False

    with patch("centml.cli.cluster.get_centml_client", return_value=context):
        yield client


def _deployment(**overrides):
    defaults = {
        "id": 123,
        "name": "test-job",
        "type": DeploymentType.JOB,
        "status": DeploymentStatus.PAUSED,
        "created_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "cluster_id": 1,
        "hardware_instance_id": 2,
        "endpoint_url": "https://jobs.example.com/test-job",
        "image_url": "registry.example.com/job:latest",
        "command": ["python", "main.py"],
        "args": ["--epochs", "1"],
        "original_command": "python main.py --epochs 1",
        "env_vars": {"ENV": "test"},
        "completions": 1,
        "parallelism": 1,
        "enable_logging": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_ls_accepts_job_type_and_displays_jobs():
    from centml.cli.cluster import ls

    deployment = _deployment()
    runner = CliRunner()

    with _patch_cluster_client() as client:
        client.get.return_value = [deployment]

        result = runner.invoke(ls, ["job"])

    assert result.exit_code == 0
    client.get.assert_called_once_with(DeploymentType.JOB)
    assert "test-job" in result.output
    assert "job" in result.output


def test_get_job_routes_to_job_api_and_displays_job_config():
    from centml.cli.cluster import get

    deployment = _deployment(status=DeploymentStatus.ACTIVE)
    hardware = SimpleNamespace(id=2, name="h100", num_gpu=8, gpu_type="H100", cost_per_hr=1200)
    runner = CliRunner()

    with _patch_cluster_client() as client:
        client.get_job.return_value = deployment
        client.get_status.return_value = SimpleNamespace(service_status=ServiceStatus.HEALTHY)
        client.get_hardware_instances.return_value = [hardware]

        result = runner.invoke(get, ["job", "123"])

    assert result.exit_code == 0
    client.get_job.assert_called_once_with(123)
    client.get_status.assert_called_once_with(123)
    assert "test-job" in result.output
    assert "ready" in result.output
    assert "Endpoint" not in result.output
    assert "https://jobs.example.com/test-job" not in result.output
    assert "registry.example.com/job:latest" in result.output
    assert "Completions" in result.output
    assert "Parallelism" in result.output
