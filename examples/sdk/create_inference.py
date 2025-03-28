import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateInferenceDeploymentRequest, UserVaultType


def main():
    with get_centml_client() as cclient:
        certs = cclient.get_user_vault(UserVaultType.CERTIFICATES)

        request = CreateInferenceDeploymentRequest(
            name="nginx",
            cluster_id=1000,
            hardware_instance_id=1000,
            image_url="nginxinc/nginx-unprivileged",
            port=8080,
            min_scale=1,
            max_scale=1,
            endpoint_certificate_authority=certs["my_cert"],
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
