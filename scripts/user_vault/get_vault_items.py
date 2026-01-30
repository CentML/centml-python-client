#!/usr/bin/env python3
"""
Script to retrieve all items from a user's vault (secrets).

This script allows you to view all secrets stored in your CentML vault,
including environment variables, SSH keys, bearer tokens, access tokens,
and certificates.
"""

from typing import Optional

import click

from centml.sdk.api import get_centml_client
from platform_api_python_client import UserVaultType


def get_vault_items(vault_type: Optional[UserVaultType] = None, search_query: Optional[str] = None):
    """Retrieve items from user's vault."""
    with get_centml_client() as client:
        response = client._api.get_all_user_vault_items_endpoint_user_vault_get(
            type=vault_type, search_query=search_query
        )
        return response.results


def display_vault_items(items, show_values: bool = False):
    """Display vault items grouped by type."""
    if not items:
        click.echo("No vault items found.")
        return

    # Group items by type
    grouped = {}
    for item in items:
        vault_type = item.type
        if vault_type not in grouped:
            grouped[vault_type] = []
        grouped[vault_type].append(item)

    click.echo(f"\nFound {len(items)} vault item(s)\n")

    for vault_type, type_items in sorted(grouped.items(), key=lambda x: x[0]):
        click.echo(f"{'=' * 50}")
        click.echo(f"Type: {vault_type} ({len(type_items)} item(s))")
        click.echo(f"{'=' * 50}")

        for item in sorted(type_items, key=lambda x: x.key):
            if show_values and item.value is not None:
                click.echo(f"  {item.key}: {item.value}")
            else:
                click.echo(f"  {item.key}")

        click.echo("")


@click.command()
@click.option(
    "--type",
    "vault_type",
    type=click.Choice([t.value for t in UserVaultType], case_sensitive=False),
    help="Filter by vault type (env_vars, ssh_keys, bearer_tokens, access_tokens, certificates)",
)
@click.option("--search", "search_query", type=str, help="Search query to filter items by key")
@click.option("--show-values", is_flag=True, default=False, help="Show vault item values")
def main(vault_type: Optional[str], search_query: Optional[str], show_values: bool):
    """Retrieve all items from user's vault (secrets).

    This script uses the centml CLI authentication,
    so make sure you are logged in to centml CLI before running this script.

    \b
    Examples:
        # Get all vault items
        python get_vault_items.py

        # Get only environment variables
        python get_vault_items.py --type env_vars

        # Search for items containing 'HF'
        python get_vault_items.py --search HF

        # Show values
        python get_vault_items.py --show-values
    """
    type_enum = UserVaultType(vault_type) if vault_type else None

    items = get_vault_items(vault_type=type_enum, search_query=search_query)

    display_vault_items(items, show_values=show_values)


if __name__ == "__main__":
    main()
