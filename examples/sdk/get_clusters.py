#!/usr/bin/env python3
"""
Script to retrieve all clusters the user has access to.

This script displays cluster information including:
- Cluster ID
- Cluster Name (Prometheus-compatible identifier)
- Display Name (human-readable)
- Region
"""

import click

from centml.sdk.api import get_centml_client


def get_clusters():
    """Retrieve all accessible clusters."""
    with get_centml_client() as client:
        return client.get_clusters().results


def display_clusters(clusters):
    """Display cluster information in a formatted table."""
    if not clusters:
        click.echo("No clusters found.")
        return

    click.echo(f"\nFound {len(clusters)} cluster(s)\n")

    for cluster in sorted(clusters, key=lambda x: x.id):
        region = cluster.region if cluster.region else "N/A"
        click.echo(f"ID:           {cluster.id}")
        click.echo(f"Cluster Name: {cluster.cluster_name}")
        click.echo(f"Display Name: {cluster.display_name}")
        click.echo(f"Region:       {region}")
        click.echo("-" * 40)


@click.command()
def main():
    """Retrieve all clusters you have access to.

    This script uses the centml CLI authentication,
    so make sure you are logged in to centml CLI before running this script.

    \b
    Examples:
        python get_clusters.py
    """
    clusters = get_clusters()
    display_clusters(clusters)


if __name__ == "__main__":
    main()
