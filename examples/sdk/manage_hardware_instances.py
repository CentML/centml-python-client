#!/usr/bin/env python3
"""
Script to manage hardware instances via the CentML SDK.

Demonstrates the hardware instance lifecycle:
- Listing hardware instances (optionally filtered by cluster)
- Creating a new hardware instance
- Deleting a hardware instance

Note: creating and deleting hardware instances require admin privileges
(PERM_ADMIN_MANAGE_HARDWARE) on your CentML organization.
"""

import click

from centml.sdk import CreateHardwareInstanceRequest
from centml.sdk.api import get_centml_client


def display_hardware_instances(instances):
    """Display hardware instance information in a formatted list."""
    if not instances:
        click.echo("No hardware instances found.")
        return

    click.echo(f"\nFound {len(instances)} hardware instance(s)\n")

    for hw in sorted(instances, key=lambda x: x.id):
        click.echo(f"ID:           {hw.id}")
        click.echo(f"Name:         {hw.name}")
        click.echo(f"Cluster ID:   {hw.cluster_id}")
        click.echo(f"GPU Type:     {hw.gpu_type}")
        click.echo(f"Num GPUs:     {hw.num_gpu}")
        click.echo(f"CPU:          {hw.cpu}")
        click.echo(f"Memory:       {hw.memory}")
        click.echo(f"Cost / hr:    {hw.cost_per_hr}")
        click.echo("-" * 40)


@click.group()
def cli():
    """Manage hardware instances.

    These commands use the centml CLI authentication, so make sure you are
    logged in to the centml CLI before running this script.
    """


@cli.command(name="list")
@click.option("--cluster-id", type=int, default=None, help="Filter to a specific cluster")
def list_instances(cluster_id):
    """List hardware instances, optionally filtered by cluster.

    \b
    Examples:
        python manage_hardware_instances.py list
        python manage_hardware_instances.py list --cluster-id 1
    """
    with get_centml_client() as client:
        instances = client.get_hardware_instances(cluster_id)
    display_hardware_instances(instances)


@cli.command()
@click.option("--cluster-id", type=int, required=True, help="Cluster the hardware belongs to")
@click.option("--name", required=True, help="Display name for the hardware instance")
@click.option("--gpu-type", required=True, help="GPU type identifier (e.g. H100, A100)")
@click.option("--num-gpu", type=int, required=True, help="Number of GPUs")
@click.option("--cpu", type=int, required=True, help="CPU in millicores")
@click.option("--memory", type=int, required=True, help="Memory in MB")
@click.option("--accelerator-resource-key", required=True, help="Kubernetes accelerator resource key")
@click.option("--accelerator-memory", type=int, required=True, help="Accelerator memory in MB")
@click.option(
    "--node-affinity-label",
    "node_affinity_labels",
    type=(str, str),
    multiple=True,
    help="Node affinity label as KEY VALUE (repeatable)",
)
def create(
    cluster_id, name, gpu_type, num_gpu, cpu, memory, accelerator_resource_key, accelerator_memory, node_affinity_labels
):
    """Create a new hardware instance (requires admin privileges).

    \b
    Examples:
        python manage_hardware_instances.py create \\
            --cluster-id 1 --name h100-8x --gpu-type H100 --num-gpu 8 \\
            --cpu 64000 --memory 128000 \\
            --accelerator-resource-key nvidia.com/gpu --accelerator-memory 80000 \\
            --node-affinity-label gpu h100
    """
    request = CreateHardwareInstanceRequest(
        cluster_id=cluster_id,
        name=name,
        gpu_type=gpu_type,
        num_gpu=num_gpu,
        cpu=cpu,
        memory=memory,
        accelerator_resource_key=accelerator_resource_key,
        node_affinity_labels=dict(node_affinity_labels),
        accelerator_memory=accelerator_memory,
    )
    with get_centml_client() as client:
        instance = client.create_hardware_instance(request)
    click.echo(f"Created hardware instance '{instance.name}' with ID {instance.id}")


@cli.command()
@click.argument("hardware_instance_id", type=int)
def delete(hardware_instance_id):
    """Delete a hardware instance by ID (requires admin privileges).

    \b
    Examples:
        python manage_hardware_instances.py delete 123
    """
    with get_centml_client() as client:
        client.delete_hardware_instance(hardware_instance_id)
    click.echo(f"Deleted hardware instance {hardware_instance_id}")


if __name__ == "__main__":
    cli()
