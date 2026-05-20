from types import SimpleNamespace

from centml.cli.cluster import _get_service_status, _get_status_error_messages
from centml.sdk import RolloutStatus, ServiceStatus


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
