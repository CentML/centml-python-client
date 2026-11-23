#!/usr/bin/env python3
"""Register your own Kubernetes cluster.

Run `centml login` or configure service-account credentials first. Registering a
cluster requires admin privileges (PERM_ADMIN_MANAGE_HARDWARE).

The cluster's API server is reached with a bearer token and its CA certificate,
base64-encoded as in a kubeconfig's `certificate-authority-data`:

    export CLUSTER_SERVER="https://<api-server>"
    export CLUSTER_BEARER_TOKEN="<service-account-token>"
    export CLUSTER_CA_CERT="<base64 PEM CA certificate>"
    export CLUSTER_STORAGE_CLASS="<storage-class>"
    python examples/sdk/register_cluster.py

Then install the cluster's components with examples/sdk/update_cluster.py.
"""

import os

from centml.sdk import ClusterCredential, ClusterRegistrationRequest
from centml.sdk.api import get_centml_client


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Set {name} before running this example.")
    return value


def main() -> None:
    request = ClusterRegistrationRequest(
        display_name=os.getenv("CLUSTER_NAME", "my-cluster"),
        storage_class=required_env("CLUSTER_STORAGE_CLASS"),
        region=os.getenv("CLUSTER_REGION"),
        credential=ClusterCredential(
            server=required_env("CLUSTER_SERVER"),
            bearer_token=required_env("CLUSTER_BEARER_TOKEN"),
            ca_cert=required_env("CLUSTER_CA_CERT"),
        ),
    )
    with get_centml_client() as client:
        cluster_id = client.register_cluster(request).id
    print(f"Registered cluster {cluster_id}")


if __name__ == "__main__":
    main()
