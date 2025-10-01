"""
Example demonstrating how to manage CServe recipes using the CentML SDK.

This example shows how to:
1. List/Get CServe recipes (read-only, no special permissions needed)
2. Update CServe recipes from platform_db.json (requires OPS admin)
3. Delete CServe recipes for specific models (requires OPS admin)

Note: Update and delete operations require platform-api-ops-client to be installed
and OPS admin permissions. Get/list operations work with just the base client.
"""

import json
from centml.sdk.ops import get_centml_ops_client


def list_recipes_example():
    """List all CServe recipes or filter by model."""
    with get_centml_ops_client() as ops_client:
        # List all recipes
        all_recipes = ops_client.get_cserve_recipes()
        print(f"Found {len(all_recipes)} recipes")

        for recipe in all_recipes[:3]:  # Show first 3
            print(f"\nModel: {recipe.model}")
            if recipe.fastest:
                print(
                    f"  Fastest - Hardware Instance: {recipe.fastest.hardware_instance_id}"
                )
            if recipe.cheapest:
                print(
                    f"  Cheapest - Hardware Instance: {recipe.cheapest.hardware_instance_id}"
                )

        # Filter by specific model
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        specific_recipes = ops_client.get_cserve_recipes(model=model_name)
        if specific_recipes:
            print(f"\nRecipe for {model_name}:")
            print(
                f"  Fastest config available: {specific_recipes[0].fastest is not None}"
            )


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
    with open("platform_db.json", "r") as f:
        platform_data = json.load(f)

    cluster_id = 1001  # Replace with your cluster ID

    with get_centml_ops_client() as ops_client:
        response = ops_client.update_cserve_recipes(
            cluster_id=cluster_id, platform_data=platform_data
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
    # Example 1: List/Get recipes (read-only, always available)
    print("=== Listing CServe Recipes ===")
    try:
        list_recipes_example()
    except Exception as e:
        print(f"Error listing recipes: {e}")

    # Example 2: Update recipes from platform_db.json (requires ops client)
    print("\n=== Updating CServe Recipes ===")
    try:
        update_recipes_example()
    except FileNotFoundError:
        print("platform_db.json not found. Skipping update example.")
    except Exception as e:
        print(f"Error updating recipes: {e}")

    # Example 3: Delete a specific model's recipe (requires ops client)
    print("\n=== Deleting CServe Recipe ===")
    try:
        delete_recipe_example()
    except Exception as e:
        print(f"Error deleting recipe: {e}")


if __name__ == "__main__":
    main()
