import warnings
from bisect import insort
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Union

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
from typing_extensions import deprecated

from centml.sdk import auth
from centml.sdk.config import settings

STATUS_V3_DEPLOYMENT_TYPES = {DeploymentType.INFERENCE_V3, DeploymentType.CSERVE_V3}

DEFAULT_LOG_PAGE_LINES = 100  # server-side default for max_lines
MAX_LOG_PAGE_LINES = 5000  # server-side ceiling for max_lines
# The server re-delivers a ~15s look-behind window on fetch-newer requests; only the
# caller's events within this generous margin of the boundary can be re-delivered.
LOG_DEDUP_RETENTION_MS = 300_000
# Merge backpressure: a pod with this many pages buffered ahead of the merge watermark
# is not fetched further until the watermark catches up. Two pages keep the merge fed
# (one page draining while the next waits) yet cap the lookahead, so a pod far ahead
# in time — typically a live pod merged with a terminated predecessor — buffers a
# couple of pages instead of its whole history.
LOG_MERGE_BUFFER_PAGES = 2


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


@dataclass
class _PodLogStream:
    """Per-pod merge state for fetch_logs: the pod's page iterator, its fetched-but-
    unreleased events awaiting the merge watermark, its watermark contribution (the
    newest buffered timestamp reading forward, the oldest reading backward), and
    whether the iterator has finished its window."""

    pages: Iterator[List[DeploymentLogEvent]]
    buffer: List[DeploymentLogEvent] = field(default_factory=list)
    frontier: int = -1
    exhausted: bool = False


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
    @deprecated("get_deployment_logs() is deprecated; use fetch_logs() instead")
    def get_deployment_logs(
        self,
        deployment_id: int,
        revision_number: int,
        pod: str,
        before: Optional[Union[list, int]] = None,
        after: Optional[Union[list, int]] = None,
        max_lines: int = DEFAULT_LOG_PAGE_LINES,
    ) -> list:
        """Deprecated: use fetch_logs() instead.

        Fetch one page of a pod's logs, oldest-first. Use get_deployment_pods() to
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
        split a millisecond, so a delivered boundary millisecond is complete unless it
        holds more than the log store's 5000-line per-query ceiling — past that the page
        carries the 5000 nearest its direction (the newest when paging older, the oldest
        when paging newer), independently of max_lines.
        """
        return self._fetch_log_page(
            deployment_id, revision_number, pod, before=before, after=after, max_lines=max_lines
        )

    # pylint: disable=R0917
    def _fetch_log_page(
        self,
        deployment_id: int,
        revision_number: int,
        pod: str,
        before: Optional[Union[list, int]] = None,
        after: Optional[Union[list, int]] = None,
        max_lines: int = DEFAULT_LOG_PAGE_LINES,
    ) -> list:
        """The page primitive behind fetch_logs and the deprecated readers — the
        contract get_deployment_logs() documents, without the deprecation warning."""
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
    @deprecated("get_deployment_logs_range() is deprecated; use fetch_logs() instead")
    def get_deployment_logs_range(
        self,
        deployment_id: int,
        revision_number: int,
        pod: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[DeploymentLogEvent]:
        """Deprecated: use fetch_logs() instead.

        Fetch every log line in [start_time, end_time] (epoch ms, inclusive; both
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
                page = self._fetch_log_page(
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

    def _iter_pod_log_pages(
        self, deployment_id: int, revision_number: int, pod: str, start_ms: int, end_time: Optional[int]
    ) -> Iterator[List[DeploymentLogEvent]]:
        """Page one pod's logs within [start_ms, end_time] oldest first, yielding each
        non-empty in-window page once. The held dedup anchor is trimmed to the server's
        re-delivery span, so state never grows with the stream."""
        held: list = []
        # after is exclusive, so start_ms - 1 admits lines at start_ms itself;
        # start_ms 0 means the whole log window — scan from the head.
        initial_boundary = max(start_ms - 1, 0)
        while True:
            anchor: Union[list, int] = _recent_anchor(held) if held else initial_boundary
            page = self._fetch_log_page(deployment_id, revision_number, pod, after=anchor, max_lines=MAX_LOG_PAGE_LINES)
            if not page:
                return
            emitted: List[DeploymentLogEvent] = []
            past_end = False
            for raw in page:
                if held and raw.id <= held[-1].id:
                    # Late arrival inside the look-behind span: keep the held window
                    # id-ordered (id order == time order) so trimming stays correct.
                    insort(held, raw, key=lambda held_event: held_event.id)
                else:
                    held.append(raw)
                # Anchoring at start_ms - 1 re-delivers the look-behind span below
                # start_ms; those ids must be held for dedup but never emitted.
                if raw.timestamp < start_ms:
                    continue
                if end_time is not None and raw.timestamp > end_time:
                    past_end = True
                    continue
                emitted.append(DeploymentLogEvent(id=raw.id, timestamp=raw.timestamp, message=raw.message, pod=pod))
            held = _recent_anchor(held)
            if emitted:
                yield emitted
            if past_end:
                return

    def _iter_pod_log_pages_backward(
        self, deployment_id: int, revision_number: int, pod: str, start_time: Optional[int], end_time: Optional[int]
    ) -> Iterator[List[DeploymentLogEvent]]:
        """Page one pod's logs within [start_time, end_time] newest page first, each
        page internally oldest-first. A backward walk visits each time region once
        and before pages carry no re-delivery span, so no dedup state is needed."""
        # before is exclusive, so end_time + 1 admits lines at end_time itself; no
        # end_time means no boundary — the server answers with the tail page.
        boundary: Optional[int] = None if end_time is None else end_time + 1
        while True:
            page = self._fetch_log_page(
                deployment_id, revision_number, pod, before=boundary, max_lines=MAX_LOG_PAGE_LINES
            )
            if not page:
                return
            emitted = [
                DeploymentLogEvent(id=raw.id, timestamp=raw.timestamp, message=raw.message, pod=pod)
                for raw in page
                if start_time is None or raw.timestamp >= start_time
            ]
            if emitted:
                yield emitted
            if start_time is not None and page[0].timestamp < start_time:
                return
            # Pages never split a millisecond, so an exclusive boundary at the oldest
            # delivered timestamp neither re-delivers nor skips.
            boundary = page[0].timestamp

    # pylint: disable=R0917
    def fetch_logs(
        self,
        deployment_id: int,
        revision_number: int,
        pod: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        newest_first: bool = False,
        chunk_size: int = DEFAULT_LOG_PAGE_LINES,
    ) -> Iterator[List[DeploymentLogEvent]]:
        """Fetch a revision's stored log lines within [start_time, end_time] (epoch
        ms, inclusive; omit either bound to leave that side unbounded), yielded
        lazily as chunks of 1..chunk_size DeploymentLogEvent — every chunk except
        possibly the last holds exactly chunk_size — each stored line at most once,
        each event carrying its pod name.

        newest_first selects only the order chunks arrive: False (the default)
        walks the window oldest chunk first, True newest chunk first. Lines inside
        every chunk are always in ascending (timestamp, id) order regardless of
        direction. The defaults read the full retained history to the present;
        newest_first=True with no bounds tails backward from the newest stored
        line; start_time alone catches up from a known point to the present.

        Exhaustion is the only termination signal: the iterator ends once the
        window is delivered, and one that yields nothing means the window holds no
        stored lines (aged out of retention, before the deployment existed, an
        unknown or not-yet-logging pod, or genuinely empty). There is no follow
        mode; tailing is a caller loop of forward fetch_logs calls with
        overlapping windows, deduplicated by event.id across calls (the README
        documents a memory-bounded recipe — cross-call dedup is the caller's job).

        pod=None merges every pod of the revision into one stream; pass a name
        from get_deployment_pods() to read a single pod. Nothing is fetched before
        the first next(), and memory stays bounded however large the window: per
        pod, at most LOG_MERGE_BUFFER_PAGES fetched pages wait in the merge and
        the forward dedup anchor is trimmed to the server's re-delivery span.

        Ordering caveats. Forward: a line the log store received late (within its
        ~15s re-delivery span) lands in a later chunk than its timestamp position —
        never duplicated, but out of order across chunks; sort by event.id where
        strict order matters. Backward: each time region is visited once, so a
        line arriving late for a region already passed is absent from that call —
        everything older than roughly the read's start minus the ingest lag is
        complete; when completeness of the newest lines matters, read forward.
        A single millisecond holding more than 5000 lines cannot be delivered
        whole: a page carries the 5000 nearest its paging direction, so a forward
        read retrieves at most 10000 of it and the middle is unreachable.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive number of lines")
        if (start_time is not None and start_time < 0) or (end_time is not None and end_time < 0):
            raise ValueError("start_time and end_time are epoch milliseconds and must not be negative")
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError("start_time must not exceed end_time")
        return self._iter_log_chunks(
            deployment_id, revision_number, pod, start_time, end_time, newest_first, chunk_size
        )

    # pylint: disable=R0917
    def _iter_log_chunks(
        self,
        deployment_id: int,
        revision_number: int,
        pod: Optional[str],
        start_time: Optional[int],
        end_time: Optional[int],
        newest_first: bool,
        chunk_size: int,
    ) -> Iterator[List[DeploymentLogEvent]]:
        """The generator behind fetch_logs, split out so fetch_logs raises its
        ValueErrors at the call site rather than at the first next()."""

        def pod_pages(name: str) -> Iterator[List[DeploymentLogEvent]]:
            if newest_first:
                return self._iter_pod_log_pages_backward(deployment_id, revision_number, name, start_time, end_time)
            return self._iter_pod_log_pages(
                deployment_id, revision_number, name, start_time if start_time is not None else 0, end_time
            )

        if pod is not None:
            batches: Iterator[List[DeploymentLogEvent]] = pod_pages(pod)
        else:
            pods = self.get_deployment_pods(deployment_id, revision_number)
            streams = {name: _PodLogStream(pages=pod_pages(name)) for name in pods}
            batches = self._merge_pod_streams(streams, newest_first)

        pending: List[DeploymentLogEvent] = []
        for batch in batches:
            if newest_first:
                # Each batch is entirely older than everything pending, so prepending
                # keeps pending ascending while chunks are cut from its newest end.
                pending[:0] = batch
                while len(pending) >= chunk_size:
                    yield pending[-chunk_size:]
                    del pending[-chunk_size:]
            else:
                for event in batch:
                    # A late arrival re-delivered inside the look-behind span can sort
                    # below lines already pending; insort keeps every chunk ascending.
                    if pending and (event.timestamp, event.id) < (pending[-1].timestamp, pending[-1].id):
                        insort(pending, event, key=lambda pending_event: (pending_event.timestamp, pending_event.id))
                    else:
                        pending.append(event)
                while len(pending) >= chunk_size:
                    yield pending[:chunk_size]
                    del pending[:chunk_size]
        if pending:
            yield pending

    def _merge_pod_streams(
        self, streams: Dict[str, _PodLogStream], newest_first: bool
    ) -> Iterator[List[DeploymentLogEvent]]:
        """Merge per-pod page iterators into (timestamp, id)-ascending batches that
        arrive oldest-first (or newest-first) across batches.

        Strict cross-pod order while any pod is still fetching: reading forward,
        release only lines at or below the least-advanced pod's frontier (its newest
        buffered timestamp); a lagging pod's buffered lines are all at or below its
        own frontier, so the minimum-frontier pod drains fully every round and the
        merge cannot deadlock. Reading backward the roles mirror: a pod's frontier
        is its oldest buffered timestamp, the watermark is the maximum frontier, and
        lines at or above it are released — the maximum-frontier pod drains fully."""
        buffer_limit = LOG_MERGE_BUFFER_PAGES * MAX_LOG_PAGE_LINES
        while True:
            for stream in streams.values():
                if stream.exhausted or len(stream.buffer) >= buffer_limit:
                    continue  # backpressure: let the merge watermark catch up before fetching more
                page = next(stream.pages, None)
                if page is None:
                    stream.exhausted = True
                    continue
                if newest_first:
                    # Backward pages are entirely older than everything buffered.
                    stream.buffer[:0] = page
                    stream.frontier = stream.buffer[0].timestamp
                else:
                    for event in page:
                        if stream.buffer and event.id <= stream.buffer[-1].id:
                            insort(stream.buffer, event, key=lambda buffered_event: buffered_event.id)
                        else:
                            stream.buffer.append(event)
                    stream.frontier = stream.buffer[-1].timestamp

            active = [stream.frontier for stream in streams.values() if not stream.exhausted]
            watermark = (max(active) if newest_first else min(active)) if active else None
            ready: List[DeploymentLogEvent] = []
            for stream in streams.values():
                if newest_first:
                    cut = len(stream.buffer)
                    while cut > 0 and (watermark is None or stream.buffer[cut - 1].timestamp >= watermark):
                        cut -= 1
                    ready += stream.buffer[cut:]
                    del stream.buffer[cut:]
                else:
                    cut = 0
                    while cut < len(stream.buffer) and (watermark is None or stream.buffer[cut].timestamp <= watermark):
                        cut += 1
                    ready += stream.buffer[:cut]
                    del stream.buffer[:cut]
            ready.sort(key=lambda event: (event.timestamp, event.id))
            if ready:
                yield ready
            if watermark is None:
                return

    @deprecated("deployment_log_session() is deprecated; use fetch_logs() instead")
    def deployment_log_session(
        self, deployment_id: int, revision_number: int, pod: str, events: Optional[list] = None
    ) -> "DeploymentLogSession":
        """Deprecated: use fetch_logs() instead.

        Stateful reader for one pod's logs that tracks fetched pages and anchors
        every request itself — see DeploymentLogSession. Seed events with logs a
        previous session (or get_deployment_logs) returned for the same pod."""
        with warnings.catch_warnings():
            # This call already warned via its own decorator; constructing the
            # (also-deprecated) session class must not warn a second time.
            warnings.simplefilter("ignore", DeprecationWarning)
            return DeploymentLogSession(self, deployment_id, revision_number, pod, events)


@deprecated("DeploymentLogSession is deprecated; use CentMLClient.fetch_logs() instead")
class DeploymentLogSession:
    """Deprecated: use CentMLClient.fetch_logs() instead.

    Maintains a contiguous, ordered window of one pod's logs across fetches.
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
        page = self._client._fetch_log_page(  # pylint: disable=protected-access
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
        delta = self._client._fetch_log_page(  # pylint: disable=protected-access
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
