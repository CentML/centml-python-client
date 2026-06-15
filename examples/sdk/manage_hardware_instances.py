#!/usr/bin/env python3
"""
Example showing how to manage hardware instances with the CentML SDK.

Covers listing, creating and deleting hardware instances. Running this script
lists the hardware instances you have access to; the create/delete helpers show
the call pattern and are not invoked automatically.

This uses the centml CLI authentication, so make sure you are logged in to the
centml CLI before running it. Creating and deleting hardware instances requires
admin privileges (PERM_ADMIN_MANAGE_HARDWARE) on your CentML organization.
"""

from centml.sdk import CreateHardwareInstanceRequest
from centml.sdk.api import get_centml_client


def list_hardware_instances():
    """List hardware instances, showing the cluster they belong to by name."""
    with get_centml_client() as client:
        clusters = {c.id: c for c in client.get_clusters().results}
        instances = client.get_hardware_instances()

    if not instances:
        print("No hardware instances found.")
        return

    print(f"\nFound {len(instances)} hardware instance(s)\n")
    for hw in sorted(instances, key=lambda x: x.id):
        cluster = clusters.get(hw.cluster_id)
        cluster_name = cluster.display_name if cluster else f"cluster {hw.cluster_id}"
        print(f"Name:      {hw.name}")
        print(f"Cluster:   {cluster_name}")
        print(f"GPU Type:  {hw.gpu_type}")
        print(f"Num GPUs:  {hw.num_gpu}")
        print(f"CPU:       {hw.cpu}")
        print(f"Memory:    {hw.memory}")
        print("-" * 40)


def create_hardware_instance():
    """Create a hardware instance (requires admin privileges)."""
    request = CreateHardwareInstanceRequest(
        cluster_id=1,
        name="h100-8x",
        gpu_type="H100",
        num_gpu=8,
        cpu=64000,
        memory=128000,
        accelerator_resource_key="nvidia.com/gpu",
        node_affinity_labels={"gpu": "h100"},
        accelerator_memory=80000,
    )
    with get_centml_client() as client:
        instance = client.create_hardware_instance(request)
    print(f"Created hardware instance '{instance.name}' with ID {instance.id}")
    return instance.id


def delete_hardware_instance(hardware_instance_id):
    """Delete a hardware instance by ID (requires admin privileges)."""
    with get_centml_client() as client:
        client.delete_hardware_instance(hardware_instance_id)
    print(f"Deleted hardware instance {hardware_instance_id}")


if __name__ == "__main__":
    list_hardware_instances()
