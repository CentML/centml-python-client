import time
import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateCServeV2DeploymentRequest, CServeV2Recipe

def get_fastest_cserve_config(cclient, model):
    return cclient.get_cserve_recipe(model=model)[0].fastest

def get_default_cserve_config(model):
    return CServeV2Recipe(model=model)

def main():
    with get_centml_client() as cclient:
        # Get fastest recipe for the Qwen model
        qwen_config = get_fastest_cserve_config(cclient, model="Qwen/Qwen2-VL-7B-Instruct")

        # Modify the recipe if necessary
        qwen_config.recipe.additional_properties["max_num_seqs"] = 512

        # Create CServeV2 deployment
        request = CreateCServeV2DeploymentRequest(
            name="qwen-fastest",
            cluster_id=cclient.get_cluster_id(qwen_config.hardware_instance_id),
            hardware_instance_id=qwen_config.hardware_instance_id,
            recipe=qwen_config.recipe,
            min_scale=1,
            max_scale=1,
            env_vars={},
        )
        response = cclient.create_cserve(request)
        print("Create deployment response: ", response)

        # Get deployment details
        deployment = cclient.get_cserve(response.id)
        print("Deployment details: ", deployment)

        '''
        # Pause the deployment
        cclient.pause(deployment.id)

        # Delete the deployment
        cclient.delete(deployment.id)
        '''

if __name__ == "__main__":
    main()
