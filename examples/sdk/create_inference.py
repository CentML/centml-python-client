import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateInferenceV3DeploymentRequest, UserVaultType


def main():
    with get_centml_client() as cclient:
        certs = cclient.get_user_vault(UserVaultType.CERTIFICATES)

        request = CreateInferenceV3DeploymentRequest(
            name="nginx",
            cluster_id=1000,
            hardware_instance_id=1000,
            image_url="nginxinc/nginx-unprivileged",
            port=8080,
            min_replicas=1,  # V3 uses min_replicas instead of min_scale
            max_replicas=3,  # V3 uses max_replicas instead of max_scale
            initial_replicas=1,  # Optional in V3 - initial number of replicas
            endpoint_certificate_authority=certs["my_cert"],
            # V3 rollout strategy parameters
            max_surge=1,  # Allow 1 extra pod during updates
            max_unavailable=0,  # Keep all pods available during updates
            healthcheck="/",
            concurrency=10,
        )
        response = cclient.create_inference(request)
        print("Create deployment response: ", response)

        ### Get deployment details (automatically detects V2 or V3)
        deployment = cclient.get_inference(response.id)
        print("Deployment details: ", deployment)

        '''
        ### Pause the deployment
        cclient.pause(deployment.id)

        ### Delete the deployment
        cclient.delete(deployment.id)
        '''

if __name__ == "__main__":
    main()
