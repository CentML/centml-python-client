#!/usr/bin/env python3
"""
Example showing how to manage cluster metadata with the CentML SDK.

Covers listing clusters (including deployment_creation_disabled) and updating
that flag via PUT /clusters/{cluster_id}/metadata. Running this script lists
clusters you can access; the update helper shows the call pattern and is not
invoked automatically.

This uses the centml CLI authentication, so make sure you are logged in to the
centml CLI before running it. Updating cluster metadata requires admin
privileges (PERM_ADMIN_MANAGE_HARDWARE) and only works on org-owned clusters
(global clusters return 404).
"""

from centml.sdk import UpdateClusterMetadataRequest
from centml.sdk.api import get_centml_client


def list_clusters():
    """List accessible clusters and whether new deployments are disabled."""
    with get_centml_client() as client:
        clusters = client.get_clusters().results

    if not clusters:
        print("No clusters found.")
        return

    print(f"\nFound {len(clusters)} cluster(s)\n")
    for cluster in sorted(clusters, key=lambda x: x.id):
        region = cluster.region if cluster.region else "N/A"
        print(f"ID:                           {cluster.id}")
        print(f"Cluster Name:                 {cluster.cluster_name}")
        print(f"Display Name:                 {cluster.display_name}")
        print(f"Region:                       {region}")
        print(f"Deployment Creation Disabled: {cluster.deployment_creation_disabled}")
        print("-" * 40)


def update_cluster_metadata(cluster_id: int, deployment_creation_disabled: bool):
    """Toggle whether new deployments can be created on an org-owned cluster."""
    request = UpdateClusterMetadataRequest(deployment_creation_disabled=deployment_creation_disabled)
    with get_centml_client() as client:
        cluster = client.update_cluster_metadata(cluster_id, request)
    print(
        f"Updated cluster {cluster.id} ({cluster.display_name}): "
        f"deployment_creation_disabled={cluster.deployment_creation_disabled}"
    )
    return cluster


if __name__ == "__main__":
    list_clusters()
    # Example (requires admin privileges on an org-owned cluster):
    # update_cluster_metadata(cluster_id=1, deployment_creation_disabled=True)
