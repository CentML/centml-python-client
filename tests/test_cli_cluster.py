from types import SimpleNamespace

from centml.cli.cluster import _get_status_error_messages


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
