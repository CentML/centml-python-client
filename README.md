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

`iter_deployment_logs()` streams a revision's logs lazily, oldest first, each line
exactly once, with bounded memory however long the log is — by default merging every
pod chronologically (each event carries its pod name). `follow=False` returns once
caught up; `follow=True` keeps tailing and picks up new pods of the revision as they
first log. `start_time` (epoch ms) bounds the beginning and `pod=` restricts to one
pod — discover names with `get_deployment_pods()` (terminated pods still within log
retention are included).

For non-streaming access: `get_deployment_logs_range()` fetches a specific time
window as a list (epoch-millisecond bounds, both optional; `pod=None` merges every
pod). A `deployment_log_session()` pages one pod statefully — `fetch_older()` toward
the beginning of history, `fetch_newer()` for only-new lines — keeping the merged,
ordered log in `.events`. The same paging is available statelessly through
`get_deployment_logs(before=..., after=...)`, anchored on events you already hold
or on a bare epoch-millisecond boundary:

```bash
python examples/sdk/get_deployment_logs.py
```

### Migrating deployment log reads from 0.5.x

`get_deployment_logs()` kept its name but not its signature: `start_time`, `end_time`,
`line_count`, `start_from_head` and `stream` are gone, and logs are read per pod. A
0.5.x call raises `TypeError` (or a validation error, if its arguments were positional)
rather than returning something wrong, so no call site fails silently.

| To | 0.5.x | 0.6.0 |
|---|---|---|
| Read a time window | `get_deployment_logs(id, rev, start_time=, end_time=)` | `get_deployment_logs_range(id, rev, start_time=, end_time=)` |
| Stream a window lazily | the same call with `stream=True` | `iter_deployment_logs(id, rev, start_time=)` |
| Take the newest lines first | `start_from_head=False` | `get_deployment_logs(id, rev, pod)`, then page with `before=` |
| Cap a page | `line_count=n` | `max_lines=n`, at most 5000 |
| Tell which pod a line came from | parse `kubernetes.pod_name` out of `message` | `event.pod` |
| Keep tailing past the window | not supported | `iter_deployment_logs(..., follow=True)` |

A whole-window read loses its envelope parsing, because `message` is now the log line
itself rather than a JSON record wrapping it:

```python
# 0.5.x
events = cclient.get_deployment_logs(DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2)
for event in events:
    record = json.loads(event["message"])
    print(record["kubernetes"]["pod_name"], record["log"])

# 0.6.0
for event in cclient.get_deployment_logs_range(DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2):
    print(event.pod, event.message)
```

A `stream=True` loop becomes an `iter_deployment_logs()` loop, which yields each page as
it arrives just as the old generator did:

```python
# 0.5.x
for event in cclient.get_deployment_logs(
    DEPLOYMENT_ID, REVISION, start_time=t1, end_time=t2, stream=True
):
    print(json.loads(event["message"])["log"])

# 0.6.0
for event in cclient.iter_deployment_logs(DEPLOYMENT_ID, REVISION, start_time=t1):
    print(event.message)
```

Two contract changes to check error handling against: a revision that does not exist now
answers 404 where the old endpoint answered 400, and a `max_lines` above 5000 is rejected
before the request leaves the client.

When paging by hand with `get_deployment_logs(after=...)`, anchor on the events you already
hold rather than on a bare timestamp. Every fetch-newer call re-delivers a short look-behind
span so late-arriving lines are not missed; an events anchor lets the SDK drop the lines you
already have, while a bare-timestamp anchor re-delivers that span undeduplicated and, once
the reader has caught up, stops advancing. `iter_deployment_logs()` handles this for you.

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
