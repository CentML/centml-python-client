from bisect import insort
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Union

import platform_api_python_client
from platform_api_python_client import (
    DeploymentType,
    DeploymentStatus,
    CreateInferenceV3DeploymentRequest,
    CreateComputeDeploymentRequest,
    CreateCServeV3DeploymentRequest,
    CreateDynamoDeploymentRequest,
    CreateJobDeploymentRequest,
    CreateHardwareInstanceRequest,
    UpdateClusterMetadataRequest,
    ApiException,
    InviteUserRequest,
    Metric,
)

from centml.sdk import auth
from centml.sdk.config import settings

STATUS_V3_DEPLOYMENT_TYPES = {DeploymentType.INFERENCE_V3, DeploymentType.CSERVE_V3}

DEFAULT_LOG_PAGE_LINES = 100  # server-side default for max_lines
MAX_LOG_PAGE_LINES = 5000  # server-side ceiling for max_lines
# The server re-delivers a ~15s look-behind window on fetch-newer requests; only the
# caller's events within this generous margin of the boundary can be re-delivered.
LOG_DEDUP_RETENTION_MS = 300_000


def _recent_anchor(events: list) -> list:
    """Trailing slice within LOG_DEDUP_RETENTION_MS of the newest event — everything
    an after anchor contributes (the exclusive boundary and the look-behind dedup
    ids), without rescanning the whole accumulated window on every page."""
    cutoff = events[-1].timestamp - LOG_DEDUP_RETENTION_MS
    first_recent = len(events)
    while first_recent > 0 and events[first_recent - 1].timestamp >= cutoff:
        first_recent -= 1
    return events[first_recent:]


@dataclass(frozen=True)
class DeploymentLogEvent:
    """One log line with its pod attached — logs_v4 events carry no pod name, so
    merged multi-pod views need the SDK to attribute each line itself."""

    id: str
    timestamp: int
    message: str
    pod: str


class CentMLClient:
    def __init__(self, api):
        self._api: platform_api_python_client.EXTERNALApi = api

    def get(self, depl_type):
        results = self._api.get_deployments_deployments_get(type=depl_type).results
        deployments = sorted(results, reverse=True, key=lambda d: d.created_at)
        return deployments

    def get_status(self, id):
        try:
            return self._api.get_deployment_status_v3_deployments_status_v3_deployment_id_get(id)
        except ApiException as e:
            if e.status in [404, 400]:
                try:
                    return self._api.get_deployment_status_deployments_status_deployment_id_get(id)
                except ApiException as v2_error:
                    raise e from v2_error
            raise

    def get_inference(self, id):
        """Get Inference deployment details - automatically handles both V2 and V3 deployments"""
        # Try V3 first (recommended), fallback to V2 if deployment is V2
        try:
            return self._api.get_inference_v3_deployment_deployments_inference_v3_deployment_id_get(id)
        except ApiException as e:
            # If V3 fails with 404 or similar, try V2
            if e.status in [404, 400]:  # Deployment might be V2 or endpoint not found
                try:
                    return self._api.get_inference_deployment_deployments_inference_deployment_id_get(id)
                except ApiException as v2_error:
                    # If both fail, raise the original V3 error as it's more likely to be the real issue
                    raise e from v2_error
            else:
                # For other errors (auth, network, etc.), raise immediately
                raise

    def get_compute(self, id):
        return self._api.get_compute_deployment_deployments_compute_deployment_id_get(id)

    def get_job(self, id):
        return self._api.get_job_deployment_deployments_job_deployment_id_get(id)

    def get_cserve(self, id):
        """Get CServe deployment details - automatically handles both V2 and V3 deployments"""
        # Try V3 first (recommended), fallback to V2 if deployment is V2
        try:
            return self._api.get_cserve_v3_deployment_deployments_cserve_v3_deployment_id_get(id)
        except ApiException as e:
            # If V3 fails with 404 or similar, try V2
            if e.status in [404, 400]:  # Deployment might be V2 or endpoint not found
                try:
                    return self._api.get_cserve_v2_deployment_deployments_cserve_v2_deployment_id_get(id)
                except ApiException as v2_error:
                    # If both fail, raise the original V3 error as it's more likely to be the real issue
                    raise e from v2_error
            else:
                # For other errors (auth, network, etc.), raise immediately
                raise

    def get_dynamo(self, id):
        return self._api.get_dynamo_deployment_deployments_dynamo_deployment_id_get(id)

    def create_inference(self, request: CreateInferenceV3DeploymentRequest):
        return self._api.create_inference_v3_deployment_deployments_inference_v3_post(request)

    def create_compute(self, request: CreateComputeDeploymentRequest):
        return self._api.create_compute_deployment_deployments_compute_post(request)

    def create_job(self, request: CreateJobDeploymentRequest):
        return self._api.create_job_deployment_deployments_job_post(request)

    def create_cserve(self, request: CreateCServeV3DeploymentRequest):
        return self._api.create_cserve_v3_deployment_deployments_cserve_v3_post(request)

    def create_dynamo(self, request: CreateDynamoDeploymentRequest):
        return self._api.create_dynamo_deployment_deployments_dynamo_post(request)

    def update_inference(self, deployment_id: int, request: CreateInferenceV3DeploymentRequest):
        return self._api.update_inference_v3_deployment_deployments_inference_v3_put(deployment_id, request)

    def update_compute(self, deployment_id: int, request: CreateComputeDeploymentRequest):
        return self._api.update_compute_deployment_deployments_compute_put(deployment_id, request)

    def update_cserve(self, deployment_id: int, request: CreateCServeV3DeploymentRequest):
        return self._api.update_cserve_v3_deployment_deployments_cserve_v3_put(deployment_id, request)

    def update_dynamo(self, deployment_id: int, request: CreateDynamoDeploymentRequest):
        return self._api.update_dynamo_deployment_deployments_dynamo_put(deployment_id, request)

    def _update_status(self, id, new_status):
        status_req = platform_api_python_client.DeploymentStatusRequest(status=new_status)
        self._api.update_deployment_status_deployments_status_deployment_id_put(id, status_req)

    def delete(self, id):
        self._update_status(id, DeploymentStatus.DELETED)

    def pause(self, id):
        self._update_status(id, DeploymentStatus.PAUSED)

    def resume(self, id):
        self._update_status(id, DeploymentStatus.ACTIVE)

    def get_clusters(self):
        return self._api.get_clusters_clusters_get()

    def update_cluster_metadata(self, cluster_id: int, request: UpdateClusterMetadataRequest):
        return self._api.update_cluster_metadata_clusters_cluster_id_metadata_put(cluster_id, request)

    def get_hardware_instances(self, cluster_id=None):
        return self._api.get_hardware_instances_hardware_instances_get(
            cluster_id=cluster_id if cluster_id else None
        ).results

    def create_hardware_instance(self, request: CreateHardwareInstanceRequest):
        return self._api.create_hardware_instance_hardware_instances_post(request)

    def delete_hardware_instance(self, hardware_instance_id: int):
        return self._api.delete_hardware_instance_hardware_instances_hardware_instance_id_delete(hardware_instance_id)

    def get_prebuilt_images(self, depl_type: DeploymentType):
        return self._api.get_prebuilt_images_prebuilt_images_get(type=depl_type)

    def get_cserve_recipe(self, model=None, hf_token=None):
        return self._api.get_cserve_recipe_deployments_cserve_recipes_get(model=model, hf_token=hf_token).results

    def get_cluster_id(self, hardware_instance_id):
        filtered_hw = list(filter(lambda h: h.id == hardware_instance_id, self.get_hardware_instances()))

        if len(filtered_hw) == 0:
            raise Exception(f"Invalid hardware instance id {hardware_instance_id}")

        return filtered_hw[0].cluster_id

    def get_user_vault(self, type):
        items = self._api.get_all_user_vault_items_endpoint_user_vault_get(type).results

        return {i.key: i.value for i in items}

    # pylint: disable=R0917
    def get_deployment_usage(
        self, id: int, metric: Metric, start_time_in_seconds: int, end_time_in_seconds: int, step: int
    ):
        return self._api.get_usage_deployments_usage_deployment_id_get(
            deployment_id=id,
            metric=metric,
            start_time_in_seconds=start_time_in_seconds,
            end_time_in_seconds=end_time_in_seconds,
            step=step,
        ).values

    def get_credits(self):
        return self._api.get_credits_credits_get()

    def initialize_user(self):
        return self._api.setup_stripe_customer_payments_setup_post()

    def invite_user(self, email: str):
        request = InviteUserRequest(email=email)
        return self._api.invite_user_organizations_invite_post(request)

    def get_capacity(self, cluster_id=None):
        return self._api.list_cluster_capacity_capacity_get(cluster_id=cluster_id).results

    def get_deployment_revisions(self, deployment_id: int):
        return self._api.get_deployment_revisions_deployments_revisions_deployment_id_get(
            deployment_id=deployment_id
        ).results

    def get_deployment_pods(self, deployment_id: int, revision_number: int) -> List[str]:
        """List pods that have logged for a deployment revision, including terminated
        pods still within log retention. A fresh deployment may return an empty list."""
        return self._api.get_deployment_pods_deployments_pods_deployment_id_revision_number_get(
            deployment_id=deployment_id, revision_number=revision_number
        ).pods

    # pylint: disable=R0917
    def get_deployment_logs(
        self,
        deployment_id: int,
        revision_number: int,
        pod: str,
        before: Optional[Union[list, int]] = None,
        after: Optional[Union[list, int]] = None,
        max_lines: int = DEFAULT_LOG_PAGE_LINES,
    ) -> list:
        """Fetch one page of a pod's logs, oldest-first. Use get_deployment_pods() to
        discover pod names and get_deployment_revisions() for the revision number.

        before and after anchor the page to events a previous call returned for the
        same pod (pass your accumulated list; only the relevant boundary is used):
          - neither: the newest page (tail).
          - before=<events>: the page strictly older than the oldest of them;
            an empty result means the beginning of history is reached.
          - after=<events>: lines strictly newer than the newest of them; an empty
            result means nothing new yet — call again later to keep tailing. Late
            lines still landing near that boundary are included on top of max_lines
            and may sort below events you already hold (order by id if that matters).
        Either anchor also accepts a bare epoch-millisecond int as the (exclusive)
        boundary itself — after=0 scans from the head of the log window; an int after
        anchor holds no event ids, so the re-delivered span at the boundary comes
        through undeduplicated. An empty anchor list raises ValueError. Pages never
        split a millisecond, so a delivered boundary millisecond is always complete.
        """
        if before is not None and after is not None:
            raise ValueError("before and after are mutually exclusive")

        fetch_newer = after is not None
        anchor = after if fetch_newer else before
        anchor_events = None
        boundary_timestamp = None
        if isinstance(anchor, int):
            boundary_timestamp = anchor
        elif anchor is not None and len(anchor) == 0:
            raise ValueError(
                "anchor events must be non-empty; omit the anchor for the tail page, "
                "or pass an epoch-ms boundary (after=0 reads from the head)"
            )
        elif anchor:
            anchor_events = anchor
            timestamps = [event.timestamp for event in anchor_events]
            boundary_timestamp = max(timestamps) if fetch_newer else min(timestamps)

        response = self._api.get_deployment_logs_v4_logs_deployment_id_revision_number_get(
            deployment_id=deployment_id,
            revision_number=revision_number,
            pod=pod,
            fetch_newer=fetch_newer,
            timestamp=boundary_timestamp,
            max_lines=max_lines,
        )
        if not fetch_newer or not anchor_events:
            return response.events

        # fetch_newer re-delivers a look-behind window at and before the boundary
        # (late-arrival protection); drop the lines the caller already holds by id.
        cutoff = max(event.timestamp for event in anchor_events) - LOG_DEDUP_RETENTION_MS
        held_event_ids = {event.id for event in anchor_events if event.timestamp >= cutoff}
        return [event for event in response.events if event.id not in held_event_ids]

    # pylint: disable=R0917
    def get_deployment_logs_range(
        self,
        deployment_id: int,
        revision_number: int,
        pod: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[DeploymentLogEvent]:
        """Fetch every log line in [start_time, end_time] (epoch ms, inclusive; both
        optional — an open end reads to the beginning or the present), oldest first.
        pod=None reads all pods of the revision and merges the streams
        chronologically; each returned event carries its pod name."""
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError("start_time must not exceed end_time")

        pods = [pod] if pod is not None else self.get_deployment_pods(deployment_id, revision_number)
        merged: List[DeploymentLogEvent] = []
        for pod_name in pods:
            events: list = []
            while True:
                # after is exclusive, so start_time - 1 admits lines at start_time itself;
                # start_time 0 (or None) means the whole window — scan from the head.
                anchor: Union[list, int] = _recent_anchor(events) if events else (start_time - 1 if start_time else 0)
                page = self.get_deployment_logs(
                    deployment_id, revision_number, pod_name, after=anchor, max_lines=MAX_LOG_PAGE_LINES
                )
                if not page:
                    break
                events += page
                if end_time is not None and page[-1].timestamp > end_time:
                    break
            merged += [
                DeploymentLogEvent(id=event.id, timestamp=event.timestamp, message=event.message, pod=pod_name)
                for event in events
                if (start_time is None or event.timestamp >= start_time)
                and (end_time is None or event.timestamp <= end_time)
            ]
        merged.sort(key=lambda event: event.id)
        return merged

    def deployment_log_session(
        self, deployment_id: int, revision_number: int, pod: str, events: Optional[list] = None
    ) -> "DeploymentLogSession":
        """Stateful reader for one pod's logs that tracks fetched pages and anchors
        every request itself — see DeploymentLogSession. Seed events with logs a
        previous session (or get_deployment_logs) returned for the same pod."""
        return DeploymentLogSession(self, deployment_id, revision_number, pod, events)


class DeploymentLogSession:
    """Maintains a contiguous, ordered window of one pod's logs across fetches.

    Every fetch is anchored on the window itself, so pages can never overlap or
    leave gaps inside it (within log retention; an undetectable gap forms if the
    session idles past retention before fetching newer lines).
    """

    # pylint: disable=R0917
    def __init__(self, client: CentMLClient, deployment_id: int, revision_number: int, pod: str, events=None):
        self._client = client
        self._deployment_id = deployment_id
        self._revision_number = revision_number
        self._pod = pod
        # Seeded events come from outside the session: canonicalize to unique ids in
        # chronological order (id order == time order at nanosecond precision).
        unique_events = {event.id: event for event in events or []}
        self._events = [unique_events[event_id] for event_id in sorted(unique_events)]

    @property
    def events(self) -> list:
        """Copy of the window fetched so far, oldest first. Complete from the beginning
        of history only once fetch_older() has returned an empty list."""
        return list(self._events)

    def fetch_older(self, max_lines: int = DEFAULT_LOG_PAGE_LINES) -> list:
        """Fetch the page older than the window and prepend it; on an empty session
        fetches the newest page (tail). Returns the page; empty list = no older
        lines exist (yet)."""
        page = self._client.get_deployment_logs(
            self._deployment_id,
            self._revision_number,
            self._pod,
            before=[self._events[0]] if self._events else None,
            max_lines=max_lines,
        )
        self._events[:0] = page
        return page

    def fetch_newer(self, max_lines: int = DEFAULT_LOG_PAGE_LINES) -> list:
        """Fetch lines newer than the window and merge them in; on an empty session
        fetches the newest page (tail) — to read from the beginning of history
        instead, loop fetch_older() until it returns an empty list. Returns only
        the new lines; empty list = nothing new yet, call again later to keep
        tailing. Rare late arrivals sort into the window below its newest lines."""
        if not self._events:
            return self.fetch_older(max_lines=max_lines)
        delta = self._client.get_deployment_logs(
            self._deployment_id,
            self._revision_number,
            self._pod,
            after=_recent_anchor(self._events),
            max_lines=max_lines,
        )
        for event in delta:
            if event.id > self._events[-1].id:
                self._events.append(event)
            else:
                # A late arrival may even precede the window's oldest line (tail page
                # cut inside the look-behind span); the server delivers that span
                # completely on top of max_lines, so the window stays contiguous.
                insort(self._events, event, key=lambda held: held.id)
        return delta


@contextmanager
def get_centml_client():
    configuration = platform_api_python_client.Configuration(
        host=settings.CENTML_PLATFORM_API_URL, access_token=auth.get_centml_token()
    )

    with platform_api_python_client.ApiClient(configuration) as api_client:
        api_instance = platform_api_python_client.EXTERNALApi(api_client)

        yield CentMLClient(api_instance)
