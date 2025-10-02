#!/usr/bin/env python3
"""
Example: Get available clusters

This example demonstrates how to retrieve cluster information
using the CentML SDK.
"""

from centml.sdk.ops import get_centml_ops_client


def main():
    """Get and display cluster information"""
    print("Retrieving cluster information...")

    with get_centml_ops_client() as ops_client:
        # Get all clusters
        clusters = ops_client.get_clusters()

        if not clusters:
            print("No clusters found.")
            return

        print(f"\nFound {len(clusters)} cluster(s):\n")

        for cluster in clusters:
            print(f"Cluster ID: {cluster.id}")
            print(f"  Display Name: {cluster.display_name}")
            print(f"  Region: {cluster.region or 'N/A'}")
            print()


if __name__ == "__main__":
    main()
