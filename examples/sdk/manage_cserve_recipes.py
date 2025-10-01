"""
Example demonstrating how to manage CServe recipes using the CentML SDK.

This example shows how to:
1. Update CServe recipes from platform_db.json
2. Delete CServe recipes for specific models

Note: This requires platform-api-ops-client to be installed and
requires OPS admin permissions.
"""

import json
from centml.sdk.ops import get_centml_ops_client


def update_recipes_example():
    """Update CServe recipes from platform_db.json file."""
    # Load platform_db.json data
    # This file should contain performance data in the format:
    # {
    #     "model_name": {
    #         "fastest": { "accelerator_type": "...", "accelerator_count": ..., ... },
    #         "cheapest": { ... },  # optional
    #         "best_value": { ... }  # optional
    #     },
    #     ...
    # }
    with open('platform_db.json', 'r') as f:
        platform_data = json.load(f)

    cluster_id = 1001  # Replace with your cluster ID

    with get_centml_ops_client() as ops_client:
        response = ops_client.update_cserve_recipes(
            cluster_id=cluster_id,
            platform_data=platform_data
        )

        print(f"Message: {response.message}")
        print(f"Processed Models: {response.processed_models}")
        if response.errors:
            print(f"Errors: {response.errors}")


def delete_recipe_example():
    """Delete CServe recipe for a specific model."""
    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    with get_centml_ops_client() as ops_client:
        ops_client.delete_cserve_recipe(model=model_name)
        print(f"Successfully deleted recipe for model: {model_name}")


def main():
    # Example 1: Update recipes from platform_db.json
    print("=== Updating CServe Recipes ===")
    try:
        update_recipes_example()
    except FileNotFoundError:
        print("platform_db.json not found. Skipping update example.")
    except Exception as e:
        print(f"Error updating recipes: {e}")

    print("\n=== Deleting CServe Recipe ===")
    # Example 2: Delete a specific model's recipe
    try:
        delete_recipe_example()
    except Exception as e:
        print(f"Error deleting recipe: {e}")


if __name__ == "__main__":
    main()

