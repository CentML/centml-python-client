#!/usr/bin/env python3
"""Install the platform components on a registered cluster.

Run `centml login` or configure service-account credentials first. Updating a
cluster requires admin privileges (PERM_ADMIN_MANAGE_HARDWARE).

    export CLUSTER_ID="<cluster-id>"
    python examples/sdk/update_cluster.py

Register the cluster first with examples/sdk/register_cluster.py.
"""

import os

from centml.sdk import UpdateClusterRequest
from centml.sdk.api import get_centml_client

# Each component maps to its Helm chart values; {} installs it with the catalog defaults.
# final_stack is required. A component left out is uninstalled, so every update must list
# all the components the cluster should keep.
COMPONENTS = {
    "kube_prometheus_stack": {},
    "istio": {},
    "cert_manager": {},
    "ingress_nginx": {},
    "keda": {},
    "argo_rollouts": {},
    "gpu_operator": {},
    "kai_scheduler": {},
    "metrics_server": {},
    "final_stack": {},
}


def main() -> None:
    cluster_id = os.getenv("CLUSTER_ID")
    if not cluster_id:
        raise SystemExit("Set CLUSTER_ID before running this example.")

    # Build the request with from_dict: the model constructors drop chart values
    # their typed models don't list.
    request = UpdateClusterRequest.from_dict({"components": COMPONENTS})
    with get_centml_client() as client:
        client.update_cluster(int(cluster_id), request)
    # The update returns without waiting for the components to roll out.
    print(f"Installing {len(COMPONENTS)} components on cluster {cluster_id}")


if __name__ == "__main__":
    main()
