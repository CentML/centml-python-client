import time
from contextlib import contextmanager
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
from centml.sdk.api import LOG_DEDUP_RETENTION_MS, LOG_MERGE_BUFFER_PAGES, CentMLClient, get_centml_client
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


def test_get_deployment_logs_zero_after_anchor_reads_from_head():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    client = CentMLClient(api)

    events = client.get_deployment_logs(123, 2, pod="pod-a", after=0)

    assert [e.id for e in events] == ["1-a"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True
    assert call.kwargs["timestamp"] == 0


def test_get_deployment_logs_rejects_empty_anchor_lists():
    api = MagicMock()
    client = CentMLClient(api)

    for kwargs in ({"before": []}, {"after": []}):
        with pytest.raises(ValueError):
            client.get_deployment_logs(123, 2, pod="pod-a", **kwargs)

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


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
        client.get_deployment_logs(123, 2, pod="pod-a", before=[_log_event("1-a", 1000)], after=0)

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
    assert calls[0].kwargs["timestamp"] == 0 and calls[0].kwargs["fetch_newer"] is True


def test_get_deployment_logs_range_merges_all_pods_by_id():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )

    def pages(**kwargs):
        if kwargs["timestamp"]:
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


def test_log_session_merges_by_production_shaped_ids():
    def _real_id(ms, suffix):
        return f"{ms * 10**6:019d}-{suffix}"

    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event(_real_id(1000, "aa"), 1000), _log_event(_real_id(2000, "bb"), 2000)),
        _log_page(_log_event(_real_id(1500, "ll"), 1500), _log_event(_real_id(3000, "cc"), 3000)),
    ]
    session = _session(api)
    session.fetch_newer()
    session.fetch_newer()

    assert [e.timestamp for e in session.events] == [1000, 1500, 2000, 3000]


def test_log_session_long_window_keeps_boundary_and_dedup_correct():
    api = MagicMock()
    old_events = [_log_event(f"{i}-x", i) for i in range(1, 4)]  # far older than the retention window
    recent = _log_event("900000-y", 900_000)
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        # Look-behind re-delivers the recent held line next to a fresh one.
        _log_page(_log_event("900000-y", 900_000), _log_event("901000-z", 901_000)),
        _log_page(_log_event("0-w", 500)),
    ]
    session = _session(api, events=old_events + [recent])

    delta = session.fetch_newer()
    assert [e.id for e in delta] == ["901000-z"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 900_000

    session.fetch_older()
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is False and call.kwargs["timestamp"] == 1


class _FakeClock:
    """Deterministic stand-in for time.monotonic/time.time/time.sleep in follow-mode tests."""

    EPOCH = 1_700_000_000.0  # wall-clock base at epoch scale, for the merge-delay watermark

    def __init__(self, max_sleeps=100):
        self.now = 0.0
        self.sleeps = 0
        self._max_sleeps = max_sleeps

    def monotonic(self):
        return self.now

    def time(self):
        return self.EPOCH + self.now

    def wall_ms(self):
        return int(self.time() * 1000)

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > self._max_sleeps:
            raise TimeoutError("test exceeded its sleep budget")
        self.now += seconds


@contextmanager
def _patched_clock(clock):
    with (
        patch("centml.sdk.api.time.monotonic", clock.monotonic),
        patch("centml.sdk.api.time.time", clock.time),
        patch("centml.sdk.api.time.sleep", clock.sleep),
    ):
        yield clock


def test_iter_deployment_logs_yields_first_page_before_fetching_the_next():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),
        _log_page(_log_event("3-c", 3000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    stream = client.iter_deployment_logs(123, 2, pod="pod-a")
    first = next(stream)

    assert first.id == "1-a" and first.pod == "pod-a"
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 1
    # The first fetch never passes an (invalid) empty anchor list: it is a bare
    # int boundary reading from the head of the log window.
    first_call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list[0]
    assert first_call.kwargs["fetch_newer"] is True and first_call.kwargs["timestamp"] == 0

    assert [e.id for e in stream] == ["2-b", "3-c"]
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 3


def test_iter_deployment_logs_terminates_when_caught_up_without_sleeping():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    with patch("centml.sdk.api.time.sleep") as sleep:
        events = list(client.iter_deployment_logs(123, 2, pod="pod-a"))

    assert [e.id for e in events] == ["1-a"]
    sleep.assert_not_called()


def _replaying_log_server(all_events, look_behind_ms=15_000):
    """Mimic the server: newer-than-boundary lines up to max_lines, plus the
    re-delivered look-behind span at and before the boundary (uncounted)."""

    def respond(**kwargs):
        boundary = kwargs["timestamp"]
        newer = [e for e in all_events if e.timestamp > boundary][: kwargs["max_lines"]]
        gray = [e for e in all_events if boundary - look_behind_ms < e.timestamp <= boundary]
        return _log_page(*sorted(gray + newer, key=lambda e: e.id))

    return respond


def test_iter_deployment_logs_does_not_livelock_on_gray_span_redelivery():
    # Regression: with a bare int boundary the re-delivered look-behind span keeps
    # every page non-empty forever (measured against dev: 9 iterations of the same
    # 7 gray lines before the naive port was declared wedged). The generator holds
    # the trailing events, so the gray span dedupes away and the stream terminates.
    all_events = [_log_event(f"{1000 + i:05d}-x", 1000 + i) for i in range(30)]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(all_events)
    client = CentMLClient(api)

    events = list(client.iter_deployment_logs(123, 2, pod="pod-a", max_lines=10))

    assert [e.id for e in events] == [e.id for e in all_events]  # every line once, in order
    # 3 data pages + 1 empty page proving catch-up; a livelock would exceed this.
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 4


def test_iter_deployment_logs_start_time_holds_gray_lines_below_the_window():
    all_events = [
        _log_event("04990-w", 4990),
        _log_event("04995-x", 4995),
        _log_event("05000-y", 5000),
        _log_event("06000-z", 6000),
    ]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(all_events)
    client = CentMLClient(api)

    events = list(client.iter_deployment_logs(123, 2, pod="pod-a", start_time=5000))

    # The look-behind lines below start_time are held for dedup but never emitted.
    assert [e.id for e in events] == ["05000-y", "06000-z"]
    first_call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list[0]
    assert first_call.kwargs["timestamp"] == 4999  # after is exclusive: admits start_time itself


def test_iter_deployment_logs_held_state_stays_within_the_dedup_window():
    step_ms = 10_000
    pages = [
        _log_page(*(_log_event(f"{(p * 100 + i) * step_ms:09d}-x", (p * 100 + i) * step_ms) for i in range(100)))
        for p in range(1, 50)
    ] + [_log_page()]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    anchor_sizes = []
    original = CentMLClient.get_deployment_logs

    def spying_get_deployment_logs(self, *args, **kwargs):
        if isinstance(kwargs.get("after"), list):
            anchor_sizes.append(len(kwargs["after"]))
        return original(self, *args, **kwargs)

    with patch.object(CentMLClient, "get_deployment_logs", spying_get_deployment_logs):
        events = list(client.iter_deployment_logs(123, 2, pod="pod-a"))

    assert len(events) == 4900
    # Held state is the trimmed dedup window, not the accumulated stream.
    assert max(anchor_sizes) <= LOG_DEDUP_RETENTION_MS // step_ms + 1


def test_iter_deployment_logs_merges_pods_by_timestamp_then_id():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )

    def pages(**kwargs):
        if kwargs["timestamp"]:
            return _log_page()
        if kwargs["pod"] == "pod-a":
            return _log_page(_log_event("1-a", 1000), _log_event("3-a", 3000))
        return _log_page(_log_event("2-b", 2000), _log_event("4-b", 4000))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    events = list(client.iter_deployment_logs(123, 2))

    assert [(e.id, e.pod) for e in events] == [("1-a", "pod-a"), ("2-b", "pod-b"), ("3-a", "pod-a"), ("4-b", "pod-b")]
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.assert_called_once()


def test_iter_deployment_logs_returns_empty_when_no_pod_has_logged():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(pods=[])
    client = CentMLClient(api)

    assert not list(client.iter_deployment_logs(123, 2))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def test_iter_deployment_logs_follow_polls_at_the_poll_interval():
    api = MagicMock()
    responses = iter([_log_page(_log_event("1-a", 1000))])
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = lambda **kwargs: next(
        responses, _log_page()
    )
    client = CentMLClient(api)
    clock = _FakeClock(max_sleeps=3)

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, pod="pod-a", follow=True, poll_interval=2.0)
        assert next(stream).id == "1-a"
        with pytest.raises(TimeoutError):
            next(stream)

    # One request per poll interval once caught up — no hot spinning.
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == clock.sleeps + 1


def test_iter_deployment_logs_follow_delivers_lines_appended_later():
    api = MagicMock()
    responses = iter([_log_page(_log_event("1-a", 1000)), _log_page(), _log_page(_log_event("2-b", 2000))])
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = lambda **kwargs: next(
        responses, _log_page()
    )
    client = CentMLClient(api)
    clock = _FakeClock()

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, pod="pod-a", follow=True)
        assert next(stream).id == "1-a"
        assert next(stream).id == "2-b"


def test_iter_deployment_logs_follow_picks_up_pods_that_appear_later():
    api = MagicMock()
    pod_lists = iter([["pod-a"]])
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.side_effect = lambda **kwargs: (
        SimpleNamespace(pods=next(pod_lists, ["pod-a", "pod-b"]))
    )

    def pages(**kwargs):
        if kwargs["pod"] == "pod-a":
            return _log_page(_log_event("1-a", 1000)) if not kwargs["timestamp"] else _log_page()
        return _log_page(_log_event("2-b", 2000)) if not kwargs["timestamp"] else _log_page()

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)
    clock = _FakeClock()

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, follow=True, poll_interval=2.0)
        assert next(stream).id == "1-a"
        appeared = next(stream)

    assert (appeared.id, appeared.pod) == ("2-b", "pod-b")
    assert api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.call_count >= 2


def test_iter_deployment_logs_follow_does_not_gate_on_a_caught_up_silent_pod():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )
    pod_a_pages = iter(
        [_log_page(_log_event("1-a", 1000), _log_event("3-a", 3000)), _log_page(_log_event("4-a", 4000))]
    )

    def pages(**kwargs):
        if kwargs["pod"] == "pod-a":
            return next(pod_a_pages, _log_page())
        return _log_page(_log_event("2-b", 2000)) if not kwargs["timestamp"] else _log_page()

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)
    clock = _FakeClock()

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, follow=True, poll_interval=2.0)
        # Round one: pod-b's frontier (2000) gates 3-a. After pod-b polls empty it is
        # caught up and stops gating, so pod-a's lines flow without any hold window.
        assert [next(stream).id for _ in range(4)] == ["1-a", "2-b", "3-a", "4-a"]


def test_iter_deployment_logs_backpressures_a_pod_far_ahead_of_the_watermark():
    # A terminated old pod gates the watermark while a live pod's history is far newer;
    # without backpressure every old-pod round would buffer another new-pod page, growing
    # the merge buffer with the live pod's whole history.
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-old", "pod-new"]
    )
    events = {
        "pod-old": [_log_event(f"old-{i:04d}", 1000 + i) for i in range(100)],
        "pod-new": [_log_event(f"new-{i:04d}", 10_000_000 + i) for i in range(100)],
    }

    def pages(**kwargs):
        newer = [e for e in events[kwargs["pod"]] if e.timestamp > (kwargs["timestamp"] or 0)]
        return _log_page(*newer[: kwargs["max_lines"]])

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    calls = {"pod-old": 0, "pod-new": 0}
    new_calls_while_old_active = []
    original = CentMLClient.get_deployment_logs

    def spying_get_deployment_logs(self, *args, **kwargs):
        calls[args[2]] += 1
        if args[2] == "pod-old":
            new_calls_while_old_active.append(calls["pod-new"])
        return original(self, *args, **kwargs)

    with patch.object(CentMLClient, "get_deployment_logs", spying_get_deployment_logs):
        yielded = list(client.iter_deployment_logs(123, 2, max_lines=10))

    assert [e.id for e in yielded] == [e.id for e in events["pod-old"]] + [e.id for e in events["pod-new"]]
    # While the old pod was still draining, the new pod was fetched at most its buffer
    # cap (LOG_MERGE_BUFFER_PAGES pages), not once per round.
    assert max(new_calls_while_old_active) <= LOG_MERGE_BUFFER_PAGES
    assert calls["pod-new"] == 11  # 10 data pages + 1 empty page, none wasted on re-polls


def test_iter_deployment_logs_follow_single_pod_appends_late_arrivals():
    # A line that reaches the log store late is re-delivered with a fresh id and yields
    # after newer lines — visible late delivery, where the CloudWatch path dropped it.
    api = MagicMock()
    responses = iter(
        [
            _log_page(*(_log_event(f"{1000 + i}-x", 1000 + i) for i in range(5))),
            _log_page(_log_event("1002-late", 1002), _log_event("2000-f", 2000)),
        ]
    )
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = lambda **kwargs: next(
        responses, _log_page()
    )
    client = CentMLClient(api)
    clock = _FakeClock()

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, pod="pod-a", follow=True)
        got = [next(stream).id for _ in range(7)]

    assert got == ["1000-x", "1001-x", "1002-x", "1003-x", "1004-x", "1002-late", "2000-f"]


def test_iter_deployment_logs_drains_lines_newer_than_the_merge_delay_on_return():
    # follow=False must drain the merge buffers unconditionally once every pod is caught
    # up; lines newer than any time-based watermark must not be silently dropped.
    recent_ms = int(time.time() * 1000) + 60_000
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )

    def pages(**kwargs):
        if kwargs["timestamp"]:
            return _log_page()
        if kwargs["pod"] == "pod-a":
            return _log_page(_log_event("1000-a", 1000), _log_event(f"{recent_ms}-a", recent_ms))
        return _log_page(_log_event("500-b", 500))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    events = list(client.iter_deployment_logs(123, 2))

    assert [e.id for e in events] == ["500-b", "1000-a", f"{recent_ms}-a"]


def test_iter_deployment_logs_follow_retires_pods_gone_from_the_pod_list():
    api = MagicMock()
    pod_lists = iter([["pod-a", "pod-b"]])
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.side_effect = lambda **kwargs: (
        SimpleNamespace(pods=next(pod_lists, ["pod-a"]))
    )
    clock = _FakeClock()
    calls = {"pod-a": 0, "pod-b": 0}
    pod_b_pages = iter([_log_page(_log_event("500-b", 500))])

    def pages(**kwargs):
        clock.now += 0.5  # requests take wall time, letting poll pacing and refresh advance
        calls[kwargs["pod"]] += 1
        if kwargs["pod"] == "pod-b":
            return next(pod_b_pages, _log_page())
        step = calls["pod-a"]
        return _log_page(_log_event(f"{10_000 + step}-a", 10_000 + step))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, follow=True, poll_interval=2.0)
        first_batch = [next(stream) for _ in range(40)]
        calls_b_after_retirement = calls["pod-b"]
        second_batch = [next(stream) for _ in range(30)]

    # pod-b left the pod list at the first refresh and was caught up: polling it stopped.
    assert calls["pod-b"] == calls_b_after_retirement
    # Its one line was delivered exactly once, and nothing else was lost or duplicated.
    ids = [e.id for e in first_batch + second_batch]
    assert ids.count("500-b") == 1 and len(set(ids)) == len(ids)


def test_iter_deployment_logs_follow_holds_steady_state_lines_for_the_merge_delay():
    # Multi-pod steady state: a residual line above a peer's frontier is released once
    # the merge delay (one poll_interval) has passed, not held indefinitely and not
    # released before its peers had a chance to interleave.
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )
    clock = _FakeClock()
    fresh_ms = clock.wall_ms() - 10
    fetch_time = {}

    def pages(**kwargs):
        if kwargs["timestamp"]:
            return _log_page()
        if kwargs["pod"] == "pod-a":
            fetch_time["fresh"] = clock.now
            return _log_page(_log_event(f"{fresh_ms - 5000}-a", fresh_ms - 5000), _log_event(f"{fresh_ms}-a", fresh_ms))
        return _log_page(_log_event(f"{fresh_ms - 6000}-b", fresh_ms - 6000))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, follow=True, poll_interval=2.0)
        assert next(stream).id == f"{fresh_ms - 6000}-b"
        assert next(stream).id == f"{fresh_ms - 5000}-a"
        released = next(stream)

    assert released.id == f"{fresh_ms}-a"
    assert clock.now - fetch_time["fresh"] >= 2.0  # held for the merge delay, then released


def test_iter_deployment_logs_follow_single_pod_releases_without_merge_delay():
    api = MagicMock()
    clock = _FakeClock()
    fresh_ms = clock.wall_ms() - 10
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event(f"{fresh_ms}-a", fresh_ms))
    ]
    client = CentMLClient(api)

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, pod="pod-a", follow=True)
        assert next(stream).id == f"{fresh_ms}-a"

    assert clock.sleeps == 0  # released in its own fetch round: no invented tail latency


def test_iter_deployment_logs_follow_paces_caught_up_pods_while_a_peer_streams():
    api = MagicMock()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(
        pods=["pod-a", "pod-b"]
    )
    clock = _FakeClock()
    calls = {"pod-a": 0, "pod-b": 0}

    def pages(**kwargs):
        clock.now += 0.5  # requests take wall time
        calls[kwargs["pod"]] += 1
        if kwargs["pod"] == "pod-b":
            return _log_page()  # permanently idle
        step = calls["pod-a"]
        return _log_page(_log_event(f"{10_000 + step}-a", 10_000 + step))

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    with _patched_clock(clock):
        stream = client.iter_deployment_logs(123, 2, follow=True, poll_interval=2.0)
        for _ in range(20):
            next(stream)

    # The idle pod is re-polled at most once per poll_interval of elapsed time, not once
    # per round of its streaming peer.
    assert calls["pod-b"] <= clock.now / 2.0 + 2
    assert calls["pod-b"] < calls["pod-a"] / 2
