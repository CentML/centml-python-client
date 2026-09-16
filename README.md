# centml-python-client
![](https://github.com/CentML/centml-python-client/actions/workflows/unit_tests.yml/badge.svg)

### Installation

To install without cloning, run the following command:
```bash
pip install git+https://github.com/CentML/centml-python-client.git@main
```

Alternatively to build from source, clone this repo then inside the project's base directory, run the following command:
```bash
pip install . 
```

### Authentication

For interactive use, authenticate once with the CLI. SDK examples reuse the stored
credentials and refresh them when needed.

```bash
centml login
python examples/sdk/validate_auth.py
```

For service-to-service use, provide both service-account environment variables:

```bash
export CENTML_SERVICE_ACCOUNT_ID="<service-account-id>"
export CENTML_SERVICE_ACCOUNT_SECRET="<service-account-secret>"
python examples/sdk/validate_auth.py
```

`CENTML_PLATFORM_API_URL` can be set when targeting a non-production API.

### Dynamo SDK example

The Dynamo example uses SDK authentication separately from the bearer token that
protects the deployed inference endpoint:

```bash
export CENTML_CLUSTER_ID="<cluster-id>"
export CENTML_HARDWARE_INSTANCE_ID="<hardware-instance-id>"
export CENTML_ENDPOINT_BEARER_TOKEN="<new-endpoint-token>"
# Required only for gated Hugging Face models:
export HF_TOKEN="<hugging-face-token>"

python examples/sdk/create_dynamo.py
```

Use `python examples/sdk/get_clusters.py` and
`python examples/sdk/manage_hardware_instances.py` to discover the required IDs.
Creating the example reserves GPU capacity and may incur usage charges. It does not
delete the deployment automatically.

### Deployment logs SDK example

`fetch_logs()` is the one way to read deployment logs: it fetches a revision's
stored log lines within a time window (`start_time`/`end_time`, epoch ms,
inclusive; omit either bound to leave that side unbounded) and yields them lazily
as chunks of up to `chunk_size` `DeploymentLogEvent` — each line at most once,
each event carrying its pod name, with bounded memory however large the window.
`newest_first` selects only the order chunks arrive: `False` (the default) walks
the window oldest chunk first, `True` newest chunk first; lines inside every chunk
are always in ascending `(timestamp, id)` order. So the bare call reads the full
retained history chronologically, `newest_first=True` starts from the newest
stored line and walks backward, and `start_time` alone catches up from a known
point to the present. By default every pod of the revision is merged into one
stream; pass `pod=` to read a single pod — discover names with
`get_deployment_pods()` (terminated pods still within log retention are included):

```python
for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION, start_time=t1_ms, end_time=t2_ms):
    for event in chunk:
        print(event.pod, event.message)
```

The iterator always terminates, and its exhaustion is the only termination signal:
one that yields nothing means the window holds no stored lines (aged out of
retention, before the deployment existed, an unknown pod, or genuinely empty).
Reading backward, note that a line the log store receives late for a time region
the walk has already passed is absent from that call: everything older than
roughly the read's start minus the ingest lag (~15s) is complete, and when
completeness of the newest lines matters, read forward. There is no follow mode:
tailing is a caller loop that re-calls a forward `fetch_logs` with a later
`start_time` and deduplicates by `event.id`:

```python
import time

OVERLAP_MS = 30_000  # covers the server's ~15s late-arrival re-delivery span
POLL_SECONDS = 2.0

seen = {}  # event id -> timestamp, trimmed to the overlap window each round
boundary = int(time.time() * 1000)  # newest timestamp seen so far
while True:
    for chunk in cclient.fetch_logs(
        DEPLOYMENT_ID, REVISION, start_time=max(boundary - OVERLAP_MS, 0), newest_first=False
    ):
        for event in chunk:
            if event.id in seen:
                continue
            seen[event.id] = event.timestamp
            boundary = max(boundary, event.timestamp)
            print(event.pod, event.message)
    cutoff = boundary - OVERLAP_MS
    seen = {event_id: ts for event_id, ts in seen.items() if ts >= cutoff}
    time.sleep(POLL_SECONDS)
```

Two properties of this loop matter. Consecutive windows overlap on purpose: the log
store may deliver a line up to ~15 seconds after its timestamp, so starting each call
`OVERLAP_MS` below the newest seen line is what keeps late arrivals from being
skipped — do not advance `start_time` past that span to avoid the duplicates. And the
dedup state is bounded: only ids inside the overlap window can come back again, so
`seen` is trimmed to that window each round and never grows with the stream.

`python examples/sdk/get_deployment_logs.py` runs a newest-first peek, a full
chronological read and this tail loop. `get_deployment_logs()`, `get_deployment_logs_range()` and
`deployment_log_session()` still work but are deprecated in favor of `fetch_logs()`
and raise a `DeprecationWarning` on use.

### Migrating deployment log reads from 0.5.x

`get_deployment_logs()` kept its name but not its signature, and is now deprecated:
`start_time`, `end_time`, `line_count`, `start_from_head` and `stream` are gone, and
`fetch_logs()` is the replacement for every read. A 0.5.x call raises `TypeError`
(or a validation error, if its arguments were positional) rather than returning
something wrong, so no call site fails silently.

| To | 0.5.x | now |
|---|---|---|
| Read a time window | `get_deployment_logs(id, rev, start_time=, end_time=)` | `fetch_logs(id, rev, start_time=, end_time=)` |
| Stream a window lazily | the same call with `stream=True` | `fetch_logs(...)` — chunks are yielded as they are fetched |
| Take the newest lines first | `start_from_head=False` | `fetch_logs(id, rev, newest_first=True)` |
| Read from the beginning | `start_from_head=True` | `fetch_logs(id, rev)` — oldest first is the default |
| Cap what one iteration hands you | `line_count=n` | `chunk_size=n` |
| Tell which pod a line came from | parse `kubernetes.pod_name` out of `message` | `event.pod` |
| Keep tailing past the window | not supported | re-call `fetch_logs` with overlapping windows (the tail loop above) |

A whole-window read loses its envelope parsing, because `message` is now the log line
itself rather than a JSON record wrapping it:

```python
# 0.5.x
events = cclient.get_deployment_logs(DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2)
for event in events:
    record = json.loads(event["message"])
    print(record["kubernetes"]["pod_name"], record["log"])

# now
for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2):
    for event in chunk:
        print(event.pod, event.message)
```

A `stream=True` loop becomes a `fetch_logs()` loop, which yields each chunk as it
arrives just as the old generator yielded pages:

```python
# 0.5.x
for event in cclient.get_deployment_logs(
    DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2, stream=True
):
    print(json.loads(event["message"])["log"])

# now
for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2):
    for event in chunk:
        print(event.message)
```

One contract change to check error handling against: a revision that does not exist
now answers 404 where the old endpoint answered 400.

The 0.6.0 readers — `get_deployment_logs()` page anchoring, `get_deployment_logs_range()`
and `deployment_log_session()` — still work but are deprecated and warn on use; the
look-behind anchoring they exposed is handled inside `fetch_logs()`:

| 0.6.0 | now |
|---|---|
| `get_deployment_logs(id, rev, pod)` — newest page of one pod | `fetch_logs(id, rev, pod=pod, newest_first=True)` and take the first chunk |
| `get_deployment_logs(id, rev, pod, after=events)` — page newer than held events | `fetch_logs(id, rev, pod=pod, start_time=boundary_ms)` (the tail loop above for repeated polling) |
| `get_deployment_logs_range(id, rev, start_time=, end_time=)` | `fetch_logs(id, rev, start_time=, end_time=)` — chunked and lazy instead of one list |
| `deployment_log_session(...).fetch_older()` loop | `fetch_logs(id, rev, pod=pod, newest_first=True)` — one iterator walks back to the start |
| `session.fetch_newer()` polling | the tail loop above |

### Un-installation

To uninstall `centml`, simply do:
```bash
pip uninstall centml
```

### CLI
Once installed, use the centml CLI tool with the following command:
```bash
centml 
```

If you want tab completion, run
```bash
source scripts/completions/completion.<shell language>
```
Shell language can be: bash, zsh, fish
(Hint: add `source /path/to/completions/completion.<shell language>` to your `~/.bashrc`, `~/.zshrc` or `~/.config/fish/completions/centml.fish`)

### Tests
To run tests, first install required packages:
```bash
pip install -r requirements-dev.txt
cd tests
```

When running on a local machine, it is recommended to run tests with the following command. This skips tests that require a GPU.
```bash
pytest --sanity
```

To run all the tests, use:
```bash
pytest
```

### Common Issues

- **`SSL` certificate on `MacOS`**

    Sometimes, you will see issues when using command like `centml cluster [CMD]`, where the output might look like:

    ```logs

    File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/urllib3/util/retry.py", line 519, in increment

    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]

    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.centml.com', port=443):

    Max retries exceeded with url: /deployments

    (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)')))
    ```

    **Solution**:
    To fix this issue, navigate to your `python` installation directory and run the `Install Certificates.command` file located there.

    For example, if you are using `python3.10`, the file path would be:
    `
    /Applications/Python 3.10/Install Certificates.command
    `
