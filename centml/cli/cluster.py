import requests
import pprint

from . import login
from .config import Config

def run(cluster_args):
    pp = pprint.PrettyPrinter(indent=4)

    centml_token = login.get_centml_token()
    headers = {'Authorization': f'Bearer {centml_token}'}

    if cluster_args.cmd == "ls":
        resp = requests.get(f"{Config.platformapi_url}/deployments", headers=headers)
        pp.pprint(resp.json()['results'])
    elif cluster_args.cmd == "deploy":
        payload = {
            "name": cluster_args.name,
            "image_url": cluster_args.image,
            "type": "inference",
            "port": cluster_args.port,
            "hardware_instance_id": "e5a5c06b-2f1c-4d41-baf2-783ffa7723dc",
            "min_replicas": 1,
            "max_replicas": 1,
            "timeout": 0,
            "healthcheck": "/",
            "env_vars": {},
            "secrets": {},
        }
        resp = requests.post(f"{Config.platformapi_url}/deployments/inference",
                json=payload,
                headers=headers)
        pp.pprint(resp)
    elif cluster_args.cmd == "delete":
        payload = {
            "status": "deleted"
        }
        resp = requests.put(f"{Config.platformapi_url}/deployments/status/{cluster_args.id}",
                json=payload,
                headers=headers)
        pp.pprint(resp.json())
    elif cluster_args.cmd == "status":
        resp = requests.get(f"{Config.platformapi_url}/deployments/status/{cluster_args.id}", headers=headers)
        pp.pprint(resp.json())
