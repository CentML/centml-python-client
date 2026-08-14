from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
    assert hasattr(platform_api_python_client.EXTERNALApi, "update_cluster_metadata_clusters_cluster_id_metadata_put")
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


def _log_event(event_id, timestamp, message="line"):
    return SimpleNamespace(id=event_id, timestamp=timestamp, message=message)


def _log_page(*events):
    return SimpleNamespace(events=list(events))


def test_generated_client_exposes_logs_v4_contract():
    assert hasattr(
        platform_api_python_client.EXTERNALApi, "get_deployment_logs_v4_logs_deployment_id_revision_number_get"
    )
    assert hasattr(
        platform_api_python_client.EXTERNALApi, "get_deployment_pods_deployments_pods_deployment_id_revision_number_get"
    )


def test_get_deployment_pods_returns_pod_names():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )
    client = CentMLClient(api)

    assert client.get_deployment_pods(123, 2) == ["pod-a", "pod-b"]

    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.assert_called_once_with(
        deployment_id=123, revision_number=2
    )


def test_get_deployment_logs_returns_tail_page_when_unanchored():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(
        _log_event("1-a", 1000), _log_event("2-b", 2000)
    )
    client = CentMLClient(api)

    events = client.get_deployment_logs(123, 2, pod="pod-a")

    assert [e.id for e in events] == ["1-a", "2-b"]
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_called_once_with(
        deployment_id=123, revision_number=2, pod="pod-a", fetch_newer=False, timestamp=None, max_lines=100
    )


def test_get_deployment_logs_before_pages_older_from_oldest_anchor():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    client = CentMLClient(api)
    held = [_log_event("2-b", 2000), _log_event("3-c", 3000)]

    events = client.get_deployment_logs(123, 2, pod="pod-a", before=held)

    assert [e.id for e in events] == ["1-a"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    # The boundary is the oldest held timestamp, passed verbatim (exclusive server-side).
    assert call.kwargs["fetch_newer"] is False
    assert call.kwargs["timestamp"] == 2000


def test_get_deployment_logs_before_empty_page_signals_beginning_of_history():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    assert client.get_deployment_logs(123, 2, pod="pod-a", before=[_log_event("1-a", 1000)]) == []


def test_get_deployment_logs_after_fetches_newer_from_newest_anchor():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("3-c", 3000))
    client = CentMLClient(api)
    held = [_log_event("1-a", 1000), _log_event("2-b", 2000)]

    events = client.get_deployment_logs(123, 2, pod="pod-a", after=held)

    assert [e.id for e in events] == ["3-c"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True
    assert call.kwargs["timestamp"] == 2000


def test_get_deployment_logs_after_drops_redelivered_lines_but_keeps_late_arrivals():
    api = MagicMock()
    # The look-behind window re-delivers 2-b (already held) plus a late arrival at 1500.
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(
        _log_event("15-l", 1500), _log_event("2-b", 2000), _log_event("3-c", 3000)
    )
    client = CentMLClient(api)
    held = [_log_event("1-a", 1000), _log_event("2-b", 2000)]

    events = client.get_deployment_logs(123, 2, pod="pod-a", after=held)

    assert [e.id for e in events] == ["15-l", "3-c"]


def test_get_deployment_logs_after_empty_anchor_reads_from_head():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    client = CentMLClient(api)

    events = client.get_deployment_logs(123, 2, pod="pod-a", after=[])

    assert [e.id for e in events] == ["1-a"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True
    assert call.kwargs["timestamp"] is None


def test_get_deployment_logs_after_empty_page_signals_nothing_new():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    assert client.get_deployment_logs(123, 2, pod="pod-a", after=[_log_event("1-a", 1000)]) == []


def test_get_deployment_logs_passes_max_lines_through():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    client.get_deployment_logs(123, 2, pod="pod-a", max_lines=7)

    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["max_lines"] == 7


def test_get_deployment_logs_rejects_before_and_after_together():
    api = MagicMock()
    client = CentMLClient(api)

    with pytest.raises(ValueError):
        client.get_deployment_logs(123, 2, pod="pod-a", before=[_log_event("1-a", 1000)], after=[])

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def _session(api, events=None):
    return CentMLClient(api).deployment_log_session(123, 2, "pod-a", events=events)


def test_log_session_first_fetch_is_tail_for_both_directions():
    for method in ("fetch_older", "fetch_newer"):
        api = MagicMock()
        api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(
            _log_event("1-a", 1000), _log_event("2-b", 2000)
        )
        session = _session(api)

        page = getattr(session, method)()

        assert [e.id for e in page] == ["1-a", "2-b"]
        assert [e.id for e in session.events] == ["1-a", "2-b"]
        call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
        assert call.kwargs["fetch_newer"] is False and call.kwargs["timestamp"] is None


def test_log_session_fetch_older_prepends_until_beginning():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("3-c", 3000), _log_event("4-d", 4000)),  # tail
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),  # older page
        _log_page(),  # beginning reached
    ]
    session = _session(api)
    session.fetch_older()

    older = session.fetch_older()
    assert [e.id for e in older] == ["1-a", "2-b"]
    assert [e.id for e in session.events] == ["1-a", "2-b", "3-c", "4-d"]
    # The older fetch anchors on the window's oldest timestamp.
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is False and call.kwargs["timestamp"] == 3000

    assert session.fetch_older() == []
    assert [e.id for e in session.events] == ["1-a", "2-b", "3-c", "4-d"]


def test_log_session_fetch_newer_merges_delta_and_sorts_late_arrivals():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),  # tail
        # Look-behind re-delivers 2-b (held) plus a late arrival at 1500 and a fresh line.
        _log_page(_log_event("15-l", 1500), _log_event("2-b", 2000), _log_event("3-c", 3000)),
    ]
    session = _session(api)
    session.fetch_newer()

    delta = session.fetch_newer()

    assert [e.id for e in delta] == ["15-l", "3-c"]
    assert [e.id for e in session.events] == ["1-a", "15-l", "2-b", "3-c"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 2000


def test_log_session_fetch_newer_empty_delta_leaves_window_unchanged():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(),
    ]
    session = _session(api)
    session.fetch_newer()

    assert session.fetch_newer() == []
    assert [e.id for e in session.events] == ["1-a"]


def test_log_session_seed_canonicalizes_and_anchors_follow_up_fetches():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    seed = [_log_event("2-b", 2000), _log_event("1-a", 1000), _log_event("2-b", 2000)]
    session = _session(api, events=seed)

    assert [e.id for e in session.events] == ["1-a", "2-b"]

    session.fetch_newer()
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 2000


def test_log_session_events_returns_a_copy():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    session = _session(api)
    session.fetch_older()

    view = session.events
    view.append(_log_event("9-z", 9000))

    assert [e.id for e in session.events] == ["1-a"]


def test_log_session_passes_max_lines_through():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    session = _session(api)

    session.fetch_older(max_lines=7)
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args.kwargs["max_lines"] == 7


def test_get_deployment_logs_accepts_timestamp_anchors():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("2-b", 2000))
    client = CentMLClient(api)

    events = client.get_deployment_logs(123, 2, pod="pod-a", after=1999)
    assert [e.id for e in events] == ["2-b"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 1999

    client.get_deployment_logs(123, 2, pod="pod-a", before=5000)
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is False and call.kwargs["timestamp"] == 5000


def test_get_deployment_logs_range_trims_to_window():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        # First page anchors at start_time-1; the look-behind may re-deliver older lines.
        _log_page(_log_event("05-x", 500), _log_event("1-a", 1000), _log_event("2-b", 2000)),
        _log_page(_log_event("3-c", 3000), _log_event("4-d", 4000)),  # newest passes end_time
    ]
    client = CentMLClient(api)

    events = client.get_deployment_logs_range(123, 2, pod="pod-a", start_time=1000, end_time=3000)

    assert [(e.id, e.pod) for e in events] == [("1-a", "pod-a"), ("2-b", "pod-a"), ("3-c", "pod-a")]
    calls = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    assert calls[0].kwargs["timestamp"] == 999
    assert calls[1].kwargs["timestamp"] == 2000
    assert len(calls) == 2  # stops once a page reaches past end_time


def test_get_deployment_logs_range_open_ended_reads_full_history():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(_log_event("2-b", 2000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    events = client.get_deployment_logs_range(123, 2, pod="pod-a")

    assert [e.id for e in events] == ["1-a", "2-b"]
    calls = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    assert calls[0].kwargs["timestamp"] is None and calls[0].kwargs["fetch_newer"] is True


def test_get_deployment_logs_range_merges_all_pods_by_id():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )

    def pages(**kwargs):
        if kwargs["timestamp"] is not None:
            return _log_page()
        if kwargs["pod"] == "pod-a":
            return _log_page(_log_event("1-a", 1000), _log_event("3-a", 3000))
        return _log_page(_log_event("2-b", 2000), _log_event("4-b", 4000))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    events = client.get_deployment_logs_range(123, 2)

    assert [(e.id, e.pod, e.message) for e in events] == [
        ("1-a", "pod-a", "line"),
        ("2-b", "pod-b", "line"),
        ("3-a", "pod-a", "line"),
        ("4-b", "pod-b", "line"),
    ]


def test_get_deployment_logs_range_returns_empty_when_no_pod_has_logged():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(pods=[])
    client = CentMLClient(api)

    assert client.get_deployment_logs_range(123, 2) == []

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def test_get_deployment_logs_range_single_millisecond_window():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000), _log_event("3-c", 3000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    events = client.get_deployment_logs_range(123, 2, pod="pod-a", start_time=2000, end_time=2000)

    assert [e.id for e in events] == ["2-b"]
    calls = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    assert calls[0].kwargs["timestamp"] == 1999


def test_get_deployment_logs_range_rejects_inverted_window():
    api = MagicMock()
    client = CentMLClient(api)

    with pytest.raises(ValueError):
        client.get_deployment_logs_range(123, 2, pod="pod-a", start_time=2000, end_time=1000)

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()
