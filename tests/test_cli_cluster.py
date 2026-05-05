from types import SimpleNamespace

import pytest

from centml.cli.cluster import DeploymentStatus, _get_status_error_messages
from centml.sdk import ApiException


def test_status_error_messages_include_revision_and_pod_messages():
    cclient = SimpleNamespace(
        get_status_v3=lambda _id: SimpleNamespace(
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
    )
    deployment = SimpleNamespace(id=123, status=DeploymentStatus.ACTIVE)

    messages = _get_status_error_messages(cclient, deployment)

    assert messages == ["revision 3: revision failed", "revision 3 / pod-a: image pull failed"]


def test_status_error_messages_do_not_repeat_duplicate_messages():
    duplicate_message = "one or more objects failed to apply"
    cclient = SimpleNamespace(
        get_status_v3=lambda _id: SimpleNamespace(
            revision_pod_details_list=[
                SimpleNamespace(
                    revision_number=None,
                    error_message=duplicate_message,
                    pod_details_list=[SimpleNamespace(name=None, error_message=duplicate_message)],
                )
            ]
        )
    )
    deployment = SimpleNamespace(id=123, status=DeploymentStatus.ACTIVE)

    assert _get_status_error_messages(cclient, deployment) == [f"revision: {duplicate_message}"]


def test_status_error_messages_fall_back_to_legacy_status_message():
    legacy_status = SimpleNamespace(error_message="legacy service failure")

    def get_status_v3(_id):
        raise ApiException(status=404)

    cclient = SimpleNamespace(get_status_v3=get_status_v3, get_status=lambda _id: legacy_status)
    deployment = SimpleNamespace(id=123, status=DeploymentStatus.ACTIVE)

    assert _get_status_error_messages(cclient, deployment) == ["legacy service failure"]


def test_status_error_messages_reraises_unexpected_v3_errors():
    def get_status_v3(_id):
        raise ApiException(status=500)

    cclient = SimpleNamespace(get_status_v3=get_status_v3)
    deployment = SimpleNamespace(id=123, status=DeploymentStatus.ACTIVE)

    with pytest.raises(ApiException):
        _get_status_error_messages(cclient, deployment)
