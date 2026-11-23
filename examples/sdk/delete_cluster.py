#!/usr/bin/env python3
"""Delete a registered cluster.

Run `centml login` or configure service-account credentials first. Deleting a
cluster requires admin privileges (PERM_ADMIN_MANAGE_HARDWARE). It uninstalls the
cluster's components and cannot be undone.

    export CLUSTER_ID="<cluster-id>"
    python examples/sdk/delete_cluster.py

List your clusters and their IDs with examples/sdk/get_clusters.py.
"""

import os

from centml.sdk.api import get_centml_client


def main() -> None:
    cluster_id = os.getenv("CLUSTER_ID")
    if not cluster_id:
        raise SystemExit("Set CLUSTER_ID before running this example.")

    with get_centml_client() as client:
        client.delete_cluster(int(cluster_id))
    print(f"Deleted cluster {cluster_id}")


if __name__ == "__main__":
    main()
