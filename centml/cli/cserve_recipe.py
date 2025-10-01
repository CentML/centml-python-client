import json
import sys
from functools import wraps

import click
from tabulate import tabulate

from centml.sdk.ops import get_centml_ops_client


def handle_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            click.echo(f"Error: {e}")
            click.echo("Please install platform-api-ops-client to use this feature.")
            return None
        except Exception as e:
            click.echo(f"Error: {e}")
            return None

    return wrapper


@click.command(help="Update CServe recipes from platform_db.json file")
@click.argument("platform_db_file", type=click.Path(exists=True))
@click.option(
    "--cluster-id",
    type=int,
    required=True,
    help="The cluster ID to associate with hardware instances",
)
@handle_exception
def update(platform_db_file, cluster_id):
    """
    Update CServe recipes from platform_db.json performance data.

    This command reads a platform_db.json file containing performance test results
    and updates the CServe recipe configurations in the database.

    Example:
        centml cserve-recipe update platform_db.json --cluster-id 1001
    """
    # Load platform_db.json file
    try:
        with open(platform_db_file, "r") as f:
            platform_data = json.load(f)
    except json.JSONDecodeError:
        click.echo(f"Error: Invalid JSON file: {platform_db_file}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error reading file: {e}")
        sys.exit(1)

    # Validate platform_data structure
    if not isinstance(platform_data, dict):
        click.echo("Error: platform_db.json should contain a dictionary of models")
        sys.exit(1)

    click.echo(f"Processing {len(platform_data)} models from {platform_db_file}...")
    click.echo(f"Target cluster ID: {cluster_id}")

    with get_centml_ops_client() as ops_client:
        response = ops_client.update_cserve_recipes(cluster_id=cluster_id, platform_data=platform_data)

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo(click.style("✓ Update Complete", fg="green", bold=True))
        click.echo("=" * 60 + "\n")

        click.echo(click.style(response.message, fg="green"))

        if response.processed_models:
            click.echo(f"\nProcessed Models ({len(response.processed_models)}):")
            for model in response.processed_models:
                click.echo(f"  ✓ {model}")

        if response.errors:
            click.echo(click.style(f"\nErrors ({len(response.errors)}):", fg="red", bold=True))
            for error in response.errors:
                click.echo(click.style(f"  ✗ {error}", fg="red"))
            sys.exit(1)


@click.command(help="Delete CServe recipe for a specific model")
@click.argument("model")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@handle_exception
def delete(model, confirm):
    """
    Delete CServe recipe configurations for a specific model.

    This will remove all recipe configurations (fastest, cheapest, best_value)
    for the specified model.

    Example:
        centml cserve-recipe delete "meta-llama/Llama-3.3-70B-Instruct"
        centml cserve-recipe delete "Qwen/Qwen3-0.6B" --confirm
    """
    if not confirm:
        if not click.confirm(f"Are you sure you want to delete recipe for model '{model}'?"):
            click.echo("Cancelled.")
            return

    with get_centml_ops_client() as ops_client:
        ops_client.delete_cserve_recipe(model=model)
        click.echo(click.style(f"✓ Successfully deleted recipe for model: {model}", fg="green"))

