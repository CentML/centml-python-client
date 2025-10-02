#!/usr/bin/env python3
"""
Example: Get available clusters

This example demonstrates how to retrieve cluster information
using the CentML SDK, with and without hardware instance details.
"""

from centml.sdk.ops import get_centml_ops_client


def main():
    """Get and display cluster information"""

    with get_centml_ops_client() as ops_client:
        # Example 1: Get clusters (basic information)
        print("=" * 60)
        print("Example 1: Get Clusters (Basic Info)")
        print("=" * 60)

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

        # Example 2: Get clusters with hardware instances
        print("\n" + "=" * 60)
        print("Example 2: Get Clusters (With Hardware Instances)")
        print("=" * 60)

        clusters_with_hw = ops_client.get_clusters(include_hardware_instances=True)

        for item in clusters_with_hw:
            cluster = item["cluster"]
            hw_instances = item["hardware_instances"]

            print(f"\nCluster: {cluster.display_name} (ID: {cluster.id})")

            if hw_instances:
                print(f"  Hardware Instances ({len(hw_instances)} available):")
                for hw in hw_instances:
                    gpu_info = (
                        f"{hw.num_accelerators}x{hw.gpu_type}"
                        if hw.num_accelerators
                        else hw.gpu_type
                    )
                    print(f"    • {hw.name} (ID: {hw.id})")
                    print(f"      GPU: {gpu_info}")
                    print(f"      CPU: {hw.cpu} cores, Memory: {hw.memory} GB")
                    print(f"      Cost: ${hw.cost_per_hr/100:.2f}/hr")
            else:
                print("  No hardware instances available")


if __name__ == "__main__":
    main()
