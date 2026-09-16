import warnings
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
from centml.sdk.api import (
    LOG_DEDUP_RETENTION_MS,
    MAX_LOG_PAGE_LINES,
    CentMLClient,
    DeploymentLogSession,
    get_centml_client,
)
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

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
        assert client.get_deployment_logs(123, 2, pod="pod-a", before=[_log_event("1-a", 1000)]) == []


def test_get_deployment_logs_after_fetches_newer_from_newest_anchor():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("3-c", 3000))
    client = CentMLClient(api)
    held = [_log_event("1-a", 1000), _log_event("2-b", 2000)]

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
        events = client.get_deployment_logs(123, 2, pod="pod-a", after=held)

    assert [e.id for e in events] == ["15-l", "3-c"]


def test_get_deployment_logs_zero_after_anchor_reads_from_head():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning):
        events = client.get_deployment_logs(123, 2, pod="pod-a", after=0)

    assert [e.id for e in events] == ["1-a"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True
    assert call.kwargs["timestamp"] == 0


def test_get_deployment_logs_rejects_empty_anchor_lists():
    api = MagicMock()
    client = CentMLClient(api)

    for kwargs in ({"before": []}, {"after": []}):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            client.get_deployment_logs(123, 2, pod="pod-a", **kwargs)

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def test_get_deployment_logs_after_empty_page_signals_nothing_new():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning):
        assert client.get_deployment_logs(123, 2, pod="pod-a", after=[_log_event("1-a", 1000)]) == []


def test_get_deployment_logs_passes_max_lines_through():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning):
        client.get_deployment_logs(123, 2, pod="pod-a", max_lines=7)

    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["max_lines"] == 7


def test_get_deployment_logs_rejects_before_and_after_together():
    api = MagicMock()
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        client.get_deployment_logs(123, 2, pod="pod-a", before=[_log_event("1-a", 1000)], after=0)

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def _session(api, events=None):
    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
        events = client.get_deployment_logs(123, 2, pod="pod-a", after=1999)
    assert [e.id for e in events] == ["2-b"]
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 1999

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
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

    with pytest.warns(DeprecationWarning):
        assert client.get_deployment_logs_range(123, 2) == []

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def test_get_deployment_logs_range_single_millisecond_window():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000), _log_event("3-c", 3000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning):
        events = client.get_deployment_logs_range(123, 2, pod="pod-a", start_time=2000, end_time=2000)

    assert [e.id for e in events] == ["2-b"]
    calls = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    assert calls[0].kwargs["timestamp"] == 1999


def test_get_deployment_logs_range_rejects_inverted_window():
    api = MagicMock()
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
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


def _flatten(chunks):
    return [event for chunk in chunks for event in chunk]


def _replaying_log_server(all_events, look_behind_ms=15_000, page_cap=None):
    """Mimic the server's fetch-newer contract: newer-than-boundary lines up to
    max_lines, plus the re-delivered look-behind span at and before the boundary
    (uncounted). page_cap simulates a server that fills pages with fewer lines
    than asked."""

    def respond(**kwargs):
        limit = min(kwargs["max_lines"], page_cap or kwargs["max_lines"])
        boundary = kwargs["timestamp"]
        newer = [e for e in all_events if e.timestamp > boundary][:limit]
        look_behind = [e for e in all_events if boundary - look_behind_ms < e.timestamp <= boundary]
        return _log_page(*sorted(look_behind + newer, key=lambda e: e.id))

    return respond


def test_fetch_logs_yields_first_chunk_before_fetching_the_next_page():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),
        _log_page(_log_event("3-c", 3000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    stream = client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=2)
    first = next(stream)

    assert [(e.id, e.pod) for e in first] == [("1-a", "pod-a"), ("2-b", "pod-a")]
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 1
    # The first fetch never passes an (invalid) empty anchor list: it is a bare
    # int boundary just below start_time (after is exclusive).
    first_call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list[0]
    assert first_call.kwargs["fetch_newer"] is True and first_call.kwargs["timestamp"] == 0

    assert [[e.id for e in chunk] for chunk in stream] == [["3-c"]]
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 3


def test_fetch_logs_requests_the_callers_chunk_size_as_max_lines():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    list(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=7))

    # chunk_size is the server page size: every request carries it as max_lines.
    calls = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    assert [call.kwargs["max_lines"] for call in calls] == [7, 7]


def test_fetch_logs_millisecond_burst_larger_than_chunk_size_arrives_whole():
    # The server never splits one millisecond across pages, so a millisecond
    # holding more than chunk_size lines comes back — and is yielded — whole.
    burst = [_log_event(f"1000-{suffix}", 1000) for suffix in "abcdefg"]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [_log_page(*burst), _log_page()]
    client = CentMLClient(api)

    chunks = list(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=3))

    assert [len(chunk) for chunk in chunks] == [7]
    assert [e.id for e in chunks[0]] == [e.id for e in burst]


def test_fetch_logs_chunk_comes_back_short_when_the_page_is_filtered():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        # The first page re-delivers the look-behind span below start_time; those
        # lines are held for dedup but not emitted, so the chunk is short.
        _log_page(
            _log_event("04990-w", 4990),
            _log_event("04995-x", 4995),
            _log_event("05000-y", 5000),
            _log_event("06000-z", 6000),
        ),
        _log_page(),
    ]
    client = CentMLClient(api)

    chunks = list(client.fetch_logs(123, 2, pod="pod-a", start_time=5000, end_time=10_000, chunk_size=4))

    assert [[e.id for e in chunk] for chunk in chunks] == [["05000-y", "06000-z"]]


def test_fetch_logs_start_time_defaults_to_now_resolved_at_the_call():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    with patch("centml.sdk.api.time.time", return_value=5000.0):
        stream = client.fetch_logs(123, 2, pod="pod-a")
    # "now" is pinned at the call, not the first next(): iterating under a later
    # clock still anchors just below the call-time millisecond (after is exclusive).
    with patch("centml.sdk.api.time.time", return_value=9000.0):
        assert next(stream) == []
    call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args
    assert call.kwargs["fetch_newer"] is True and call.kwargs["timestamp"] == 5_000_000 - 1


def test_fetch_logs_end_time_truncates_the_window_and_stops_fetching():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(
        [_log_event(f"{1000 * i}-x", 1000 * i) for i in range(1, 9)]
    )
    client = CentMLClient(api)

    events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=2000, end_time=5000))

    assert [e.id for e in events] == ["2000-x", "3000-x", "4000-x", "5000-x"]
    # The page carrying lines past end_time already proves the window is complete:
    # no further request is issued.
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 1


def test_fetch_logs_bounded_read_terminates_when_caught_up():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    # end_time still in the future: an empty page means the store is caught up,
    # and with an end bound set that terminates the read.
    events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000))

    assert [e.id for e in events] == ["1-a"]
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 2


def test_fetch_logs_open_ended_yields_empty_chunks_and_resumes_without_duplicates():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),
        _log_page(),  # caught up: "nothing new yet"
        _log_page(),  # still nothing
        _log_page(_log_event("2-b", 2000), _log_event("3-c", 3000)),  # look-behind re-delivers 2-b
        _log_page(),
    ]
    client = CentMLClient(api)

    stream = client.fetch_logs(123, 2, pod="pod-a", start_time=1)

    # Without end_time the generator never returns: an empty chunk is the
    # "nothing new yet" signal and the caller decides when to sleep or break.
    assert [e.id for e in next(stream)] == ["1-a", "2-b"]
    assert next(stream) == []
    assert next(stream) == []
    # It resumes delivering once new lines are stored, and the line re-delivered
    # inside the look-behind span never crosses the empty-chunk boundary twice.
    assert [e.id for e in next(stream)] == ["3-c"]
    assert next(stream) == []


def test_fetch_logs_does_not_livelock_on_look_behind_redelivery():
    # A bare int boundary alone would never terminate: the re-delivered look-behind
    # span keeps every page non-empty and the boundary never advances past it. The
    # held anchor dedupes the span away, so the bounded read terminates.
    all_events = [_log_event(f"{1000 + i:05d}-x", 1000 + i) for i in range(30)]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(
        all_events, page_cap=10
    )
    client = CentMLClient(api)

    events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=100_000))

    assert [e.id for e in events] == [e.id for e in all_events]  # every line once, in order
    # 3 data pages + 1 empty page proving catch-up; a livelock would exceed this.
    assert api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_count == 4


def test_fetch_logs_start_time_holds_look_behind_lines_below_the_window():
    all_events = [
        _log_event("04990-w", 4990),
        _log_event("04995-x", 4995),
        _log_event("05000-y", 5000),
        _log_event("06000-z", 6000),
    ]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(all_events)
    client = CentMLClient(api)

    events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=5000, end_time=100_000))

    # The look-behind lines below start_time are held for dedup but never emitted.
    assert [e.id for e in events] == ["05000-y", "06000-z"]
    first_call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list[0]
    assert first_call.kwargs["timestamp"] == 4999  # after is exclusive: admits start_time itself


def test_fetch_logs_held_state_stays_within_the_dedup_window():
    step_ms = 10_000
    pages = [
        _log_page(*(_log_event(f"{(p * 100 + i) * step_ms:09d}-x", (p * 100 + i) * step_ms) for i in range(100)))
        for p in range(1, 50)
    ] + [_log_page()]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = pages
    client = CentMLClient(api)

    anchor_sizes = []
    original = CentMLClient._fetch_log_page

    def spying_fetch_log_page(self, *args, **kwargs):
        if isinstance(kwargs.get("after"), list):
            anchor_sizes.append(len(kwargs["after"]))
        return original(self, *args, **kwargs)

    with patch.object(CentMLClient, "_fetch_log_page", spying_fetch_log_page):
        events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10**12))

    assert len(events) == 4900
    # Held state is the trimmed dedup window, not the accumulated stream.
    assert max(anchor_sizes) <= LOG_DEDUP_RETENTION_MS // step_ms + 1


def test_fetch_logs_single_millisecond_window():
    all_events = [_log_event(f"{1000 * i:05d}-x", 1000 * i) for i in range(1, 4)]
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = _replaying_log_server(all_events)
    client = CentMLClient(api)

    events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=2000, end_time=2000))

    assert [e.id for e in events] == ["02000-x"]


def test_fetch_logs_late_arrival_lands_in_a_later_chunk_still_ascending():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("01000-a", 1000), _log_event("02000-b", 2000), _log_event("03000-c", 3000)),
        # A late arrival re-delivered inside the look-behind span lands in the
        # chunk of the page that carried it, below that page's fresh lines.
        _log_page(_log_event("02500-l", 2500), _log_event("04000-d", 4000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    chunks = list(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=3))

    assert [[e.id for e in chunk] for chunk in chunks] == [["01000-a", "02000-b", "03000-c"], ["02500-l", "04000-d"]]


def test_fetch_logs_chunk_stays_ascending_when_the_server_ties_on_timestamp():
    # The server sorts a page by nanosecond timestamp only, so two lines sharing
    # one nanosecond can arrive in either id order; the chunk must still come out
    # ascending in (timestamp, id).
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1000-b", 1000), _log_event("1000-a", 1000), _log_event("2000-c", 2000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    chunks = list(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=5))

    assert [[e.id for e in chunk] for chunk in chunks] == [["1000-a", "1000-b", "2000-c"]]


def test_fetch_logs_validates_eagerly_at_the_call_not_the_first_next():
    api = MagicMock()
    client = CentMLClient(api)

    # Each ValueError is raised by the call itself — never deferred to next() —
    # so a stored or passed-around iterator cannot surface it far from the bad call.
    for kwargs in (
        {"start_time": 2000, "end_time": 1000},
        {"start_time": -1},
        {"end_time": -1},
        {"chunk_size": 0},
        # chunk_size is the on-the-wire page size, so it inherits the server's
        # ceiling — rejected here, not by a generated-model pydantic error.
        {"chunk_size": MAX_LOG_PAGE_LINES + 1},
    ):
        with pytest.raises(ValueError):
            client.fetch_logs(123, 2, pod="pod-a", **kwargs)

    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()


def test_fetch_logs_accepts_both_ends_of_the_chunk_size_range():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    for chunk_size in (1, MAX_LOG_PAGE_LINES):
        list(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=chunk_size))

    sent = [
        call.kwargs["max_lines"]
        for call in api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list
    ]
    assert sent == [1, MAX_LOG_PAGE_LINES]


def test_fetch_logs_start_time_zero_clamps_the_boundary():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    client = CentMLClient(api)

    list(client.fetch_logs(123, 2, pod="pod-a", start_time=0, end_time=10_000))

    # after is exclusive, but the server rejects a negative boundary, so start_time 0
    # clamps to 0 — which already admits every stored line.
    first_call = api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.call_args_list[0]
    assert first_call.kwargs["timestamp"] == 0


def test_fetch_logs_failed_page_leaves_delivered_chunks_whole():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000), _log_event("2-b", 2000)),
        ApiException(status=503),
    ]
    client = CentMLClient(api)

    stream = client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000, chunk_size=2)

    assert [e.id for e in next(stream)] == ["1-a", "2-b"]
    with pytest.raises(ApiException):
        next(stream)
    # The failure ends the iterator, as it would any generator; the caller resumes
    # with a new fetch_logs anchored on the last event it holds.
    with pytest.raises(StopIteration):
        next(stream)


def test_fetch_logs_is_lazy_until_the_first_next():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page(_log_event("1-a", 1000))
    client = CentMLClient(api)

    stream = client.fetch_logs(123, 2, pod="pod-a", start_time=1, chunk_size=1)

    # Nothing is fetched before the first next().
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_not_called()

    next(stream)
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.assert_called_once()


def test_fetch_logs_does_not_warn_on_its_own_internal_calls():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.side_effect = [
        _log_page(_log_event("1-a", 1000)),
        _log_page(),
    ]
    client = CentMLClient(api)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        events = _flatten(client.fetch_logs(123, 2, pod="pod-a", start_time=1, end_time=10_000))

    assert len(events) == 1


def test_deprecated_log_readers_warn_and_name_the_replacement():
    api = MagicMock()
    api.get_deployment_logs_v4_logs_deployment_id_revision_number_get.return_value = _log_page()
    api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get.return_value = SimpleNamespace(pods=[])
    client = CentMLClient(api)

    with pytest.warns(DeprecationWarning, match="fetch_logs"):
        client.get_deployment_logs(123, 2, pod="pod-a")
    with pytest.warns(DeprecationWarning, match="fetch_logs"):
        client.get_deployment_logs_range(123, 2)
    with pytest.warns(DeprecationWarning, match="fetch_logs") as caught:
        session = client.deployment_log_session(123, 2, "pod-a")
    # Exactly one warning: the method's own, not a second from constructing the
    # (also-deprecated) session class inside it.
    assert len(caught) == 1
    with pytest.warns(DeprecationWarning, match="fetch_logs"):
        DeploymentLogSession(client, 123, 2, "pod-a")

    # The deprecated paths still work, and a constructed session fetches without
    # re-warning on the SDK's own internal page calls.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert session.fetch_older() == []
