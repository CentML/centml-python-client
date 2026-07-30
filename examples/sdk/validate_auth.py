#!/usr/bin/env python3
"""Validate CentML SDK authentication with a read-only API request."""

from centml.sdk import ApiException
from centml.sdk.api import get_centml_client


def validate_auth():
    """Return accessible clusters after validating the configured credentials."""
    with get_centml_client() as client:
        return client.get_clusters().results


def main():
    try:
        clusters = validate_auth()
    except SystemExit as error:
        raise SystemExit(
            "SDK authentication failed. Run `centml login` or set both "
            "CENTML_SERVICE_ACCOUNT_ID and CENTML_SERVICE_ACCOUNT_SECRET."
        ) from error
    except ApiException as error:
        if error.status in (401, 403):
            raise SystemExit(
                "The Platform API rejected the SDK credentials. Run `centml login` "
                "or verify the service-account credentials and API URL."
            ) from error
        raise

    print("SDK authentication succeeded.")
    print(f"Accessible clusters: {len(clusters)}")


if __name__ == "__main__":
    main()
