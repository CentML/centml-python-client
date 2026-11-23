from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from centml.cli.cluster import _get_ready_status, _get_service_status, _get_status_error_messages
from centml.sdk import DeploymentStatus, DeploymentType, HardwareInstanceResponse, RolloutStatus, ServiceStatus


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


def test_ready_status_supports_completed_service_status():
    deployment = SimpleNamespace(status=DeploymentStatus.ACTIVE)

    assert "completed" in _get_ready_status(deployment, ServiceStatus.COMPLETED)


def test_ready_status_supports_cleaned_up_service_status():
    deployment = SimpleNamespace(status=DeploymentStatus.ACTIVE)

    assert "cleanedUp" in _get_ready_status(deployment, ServiceStatus.CLEANEDUP)


def test_ready_status_supports_failed_service_status():
    deployment = SimpleNamespace(status=DeploymentStatus.ACTIVE)

    assert "failed" in _get_ready_status(deployment, ServiceStatus.FAILED)


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


@pytest.mark.parametrize("command,args", [("get", ["cserve", "123"]), ("ls", ["cserve"])])
def test_retired_cserve_type_is_rejected(command, args):
    from centml.cli import cluster

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(getattr(cluster, command), args)

    assert result.exit_code == 2
    assert "Invalid value for" in result.output
    client.get.assert_not_called()
    client.get_cserve.assert_not_called()


def test_ls_displays_dynamo_deployments():
    from centml.cli.cluster import ls

    with _patch_cluster_client() as client:
        client.get.return_value = [_deployment(type=DeploymentType.DYNAMO, name="test-dynamo")]
        result = CliRunner().invoke(ls)

    assert result.exit_code == 0
    assert "test-dynamo" in result.output
    assert "dynamo" in result.output


@pytest.mark.parametrize(
    "deployment_type", ["ccl_compute", "ccl_inference", "ccl_job", "ccl_dynamo", "future_type", "cserve_v3"]
)
def test_ls_hides_unsupported_deployment_types(deployment_type):
    from centml.cli.cluster import ls

    with _patch_cluster_client() as client:
        client.get.return_value = [
            _deployment(type=DeploymentType(deployment_type), name="hidden-deployment"),
            _deployment(type=DeploymentType.DYNAMO, name="visible-dynamo"),
        ]
        result = CliRunner().invoke(ls)

    assert result.exit_code == 0
    assert "hidden-deployment" not in result.output
    assert deployment_type not in result.output
    assert "visible-dynamo" in result.output


def test_get_job_routes_to_job_api_and_displays_job_config():
    from centml.cli.cluster import get

    deployment = _deployment(status=DeploymentStatus.ACTIVE)
    hardware = HardwareInstanceResponse(
        id=2, name="h100", gpu_type="H100", num_gpu=8, cpu=96, memory=1024, cost_per_hr=0, cluster_id=1
    )
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
    assert "h100 (8x H100)" in result.output
    assert "credits/hr" not in result.output


def test_register_sends_registration_request_and_prints_cluster_id():
    from centml.cli.cluster import register_cluster
    from centml.sdk import CloudProvider

    with _patch_cluster_client() as client:
        client.register_cluster.return_value = SimpleNamespace(id=7)

        result = CliRunner().invoke(
            register_cluster,
            [
                "--name",
                "my-cluster",
                "--storage-class",
                "standard",
                "--server",
                "https://k8s.example.com",
                "--ca-cert",
                "Y2E=",
                "--region",
                "us-east-1",
                "--cloud-provider",
                "aws",
            ],
            input="secret-token\n",
        )

    assert result.exit_code == 0, result.output
    assert "Registered cluster 7" in result.output
    assert "secret-token" not in result.output
    request = client.register_cluster.call_args.args[0]
    assert request.display_name == "my-cluster"
    assert request.storage_class == "standard"
    assert request.region == "us-east-1"
    assert request.cloud_provider == CloudProvider.AWS
    assert request.credential.server == "https://k8s.example.com"
    assert request.credential.ca_cert == "Y2E="
    assert request.credential.bearer_token == "secret-token"


@pytest.mark.parametrize(
    "contents",
    [
        "components:\n"
        "  final_stack: {}\n"
        "  ingress_nginx:\n"
        "    ingress-nginx:\n"
        "      controller:\n"
        "        podLabels:\n"
        "          sidecar.istio.io/inject: 'true'\n"
        "        untypedKey: 1\n",
        '{"components": {"final_stack": {}, "ingress_nginx": {"ingress-nginx": {"controller": '
        '{"podLabels": {"sidecar.istio.io/inject": "true"}, "untypedKey": 1}}}}}',
    ],
    ids=["yaml", "json"],
)
def test_update_cluster_sends_request_from_file_keeping_untyped_chart_keys(tmp_path, contents):
    from centml.cli.cluster import update_cluster

    request_file = tmp_path / "request"
    request_file.write_text(contents)

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(update_cluster, ["7", str(request_file)])

    assert result.exit_code == 0, result.output
    cluster_id, request = client.update_cluster.call_args.args
    assert cluster_id == 7
    components = request.to_dict()["components"]
    assert components["final_stack"] == {}
    controller = components["ingress_nginx"]["ingress-nginx"]["controller"]
    assert controller["podLabels"] == {"sidecar.istio.io/inject": "true"}
    assert controller["untypedKey"] == 1


@pytest.mark.parametrize(
    "contents, error",
    [
        ("components:\n  final_stack: {}\n  ingress-nginx: {}\n", "unknown component(s) ingress-nginx"),
        ("final_stack: {}\n", 'single "components" key'),
        ("component:\n  final_stack: {}\n", 'single "components" key'),
        ("components:\n  final_stack: {}\nextra: 1\n", 'single "components" key'),
        ("components: []\n", '"components" must be a mapping'),
        ("components:\n  istio: {}\n", "final_stack"),
        ("components: [\n", "not valid YAML or JSON"),
    ],
    ids=[
        "unknown-component",
        "no-wrapper",
        "misspelled-wrapper",
        "extra-key",
        "not-mapping",
        "no-final-stack",
        "bad-yaml",
    ],
)
def test_update_cluster_rejects_invalid_request_file(tmp_path, contents, error):
    from centml.cli.cluster import update_cluster

    request_file = tmp_path / "request.yaml"
    request_file.write_text(contents)

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(update_cluster, ["7", str(request_file)])

    assert result.exit_code == 2
    assert "Invalid value for REQUEST_FILE" in result.output
    assert error in result.output
    client.update_cluster.assert_not_called()


def test_list_clusters_shows_clusters_sorted_by_id():
    from centml.cli.cluster import list_clusters

    with _patch_cluster_client() as client:
        client.get_clusters.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(id=9, display_name="second", region=None),
                SimpleNamespace(id=3, display_name="first", region="us-east-1"),
            ]
        )

        result = CliRunner().invoke(list_clusters)

    assert result.exit_code == 0, result.output
    assert "us-east-1" in result.output
    assert result.output.index("first") < result.output.index("second")


def test_delete_cluster_deletes_after_confirmation():
    from centml.cli.cluster import delete_cluster

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(delete_cluster, ["7"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "Deleted cluster 7" in result.output
    client.delete_cluster.assert_called_once_with(7)


def test_delete_cluster_skips_prompt_with_yes():
    from centml.cli.cluster import delete_cluster

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(delete_cluster, ["7", "--yes"])

    assert result.exit_code == 0, result.output
    client.delete_cluster.assert_called_once_with(7)


def test_delete_cluster_does_nothing_when_declined():
    from centml.cli.cluster import delete_cluster

    with _patch_cluster_client() as client:
        result = CliRunner().invoke(delete_cluster, ["7"], input="n\n")

    assert result.exit_code == 1
    client.delete_cluster.assert_not_called()
