import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateInferenceDeploymentRequest, UserVaultType



def main():
    with get_centml_client() as cclient:
        token = cclient.get_user_vault(UserVaultType.BEARER_TOKENS)
        request = CreateInferenceDeploymentRequest(
            name="vllm",
            cluster_id=1000,
            hardware_instance_id=1000,
            image_url="vllm",
            port=8080,
            min_scale=1,
            max_scale=1,
            endpoint_bearer_token=token["general-inference"], #token must exist in vault
        )
        response = cclient.create_inference(request)
        print("Create deployment response: ", response)

        ### Get deployment details
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
