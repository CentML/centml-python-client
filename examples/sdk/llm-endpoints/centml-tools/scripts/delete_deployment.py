import json
import sys
from centml.sdk.api import get_centml_client

def load_deployment_name(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            name = config.get("deployment_name", "").strip()
            if not name:
                raise ValueError("Missing 'deployment_name' in config.")
            return name
    except Exception as e:
        print(f"❌ Failed to read config file: {e}")
        sys.exit(1)

def delete_if_exists(cclient, deployment_name):
    print(f"\n📋 Searching for deployment named: '{deployment_name}' (case-insensitive)")

    try:
        deployments = cclient._api.get_deployments_deployments_get().results
    except Exception as e:
        print(f"❌ Failed to list deployments: {e}")
        sys.exit(1)

    matched = next(
        (d for d in deployments if getattr(d, "name", "").strip().lower() == deployment_name.lower()),
        None
    )

    if not matched:
        print("ℹ️ No matching deployment found.")
        return

    print(f"🗑️ Deleting deployment '{matched.name}' (id={matched.id})...")
    try:
        cclient.delete(matched.id)
        print("✅ Deployment deleted successfully.")
    except Exception as e:
        print(f"❌ Failed to delete deployment: {e}")
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 delete_deployment.py <config_file.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    with get_centml_client() as cclient:
        deployment_name = load_deployment_name(config_path)
        delete_if_exists(cclient, deployment_name)

if __name__ == "__main__":
    main()

