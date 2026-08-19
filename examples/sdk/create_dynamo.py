#!/usr/bin/env python3
"""Create and inspect a fixed-replica NVIDIA Dynamo deployment.

Run `centml login` or configure service-account credentials first. This example
creates a billable GPU deployment and intentionally leaves cleanup to the user.
"""

import os

from centml.sdk import CreateDynamoDeploymentRequest
from centml.sdk.api import get_centml_client


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Set {name} before running this example.")
    return value


def build_request() -> CreateDynamoDeploymentRequest:
    hf_token = os.getenv("HF_TOKEN")
    return CreateDynamoDeploymentRequest(
        name=os.getenv("CENTML_DEPLOYMENT_NAME", "qwen-dynamo"),
        cluster_id=int(required_env("CENTML_CLUSTER_ID")),
        hardware_instance_id=int(required_env("CENTML_HARDWARE_INSTANCE_ID")),
        model=os.getenv("DYNAMO_MODEL", "Qwen/Qwen3-0.6B"),
        min_replicas=1,
        max_replicas=1,
        hf_token=hf_token or None,
        endpoint_bearer_token=required_env("CENTML_ENDPOINT_BEARER_TOKEN"),
    )


def validate_target(client, request: CreateDynamoDeploymentRequest) -> None:
    clusters = {cluster.id for cluster in client.get_clusters().results}
    if request.cluster_id not in clusters:
        raise SystemExit(f"Cluster {request.cluster_id} is not accessible to the authenticated identity.")

    hardware_instances = {hardware.id for hardware in client.get_hardware_instances(cluster_id=request.cluster_id)}
    if request.hardware_instance_id not in hardware_instances:
        raise SystemExit(
            f"Hardware instance {request.hardware_instance_id} is not available in cluster {request.cluster_id}."
        )


def main():
    request = build_request()

    with get_centml_client() as client:
        validate_target(client, request)

        response = client.create_dynamo(request)
        deployment = client.get_dynamo(response.id)

        # Lifecycle helpers are type-independent:
        # client.pause(deployment.id)
        # client.resume(deployment.id)
        # client.delete(deployment.id)
        #
        # Scale replicas in place (no new revision / no rolling update):
        # Changing only min_replicas and/or max_replicas on the same Create*
        # request updates the selected revision in place; revision_number stays put.
        # Any other field change still creates a new revision.
        # request.min_replicas = 2
        # request.max_replicas = 4
        # client.update_dynamo(deployment.id, request)
        # scaled = client.get_dynamo(deployment.id)
        # print(f"Scaled to {scaled.min_replicas}-{scaled.max_replicas}; revision {scaled.revision_number}")

    print(f"Created Dynamo deployment {deployment.id}: {deployment.endpoint_url}")
    print(f"Model: {deployment.model}")
    print(f"Replicas: {deployment.min_replicas}-{deployment.max_replicas}")
    print(f"Status: {deployment.status.value}")


if __name__ == "__main__":
    main()
