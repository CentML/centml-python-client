import time
import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateCServeV2DeploymentRequest

with get_centml_client() as cclient:
    # Get fastest recipe for the Qwen model
    fastest = cclient.get_cserve_recipe(model="Qwen/Qwen2-VL-7B-Instruct")[0].fastest

    # Modify the recipe if necessary
    fastest.recipe.additional_properties["max_num_seqs"] = 512

    # Create CServeV2 deployment
    request = CreateCServeV2DeploymentRequest(
        name="qwen-fastest",
        cluster_id=cclient.get_cluster_id(fastest.hardware_instance_id),
        hardware_instance_id=fastest.hardware_instance_id,
        recipe=fastest.recipe,
        min_scale=1,
        max_scale=1,
        env_vars={},
    )
    response = cclient.create_cserve(request)
    print("Create deployment response: ", response)

    # Get deployment details
    deployment = cclient.get_cserve(response.id)
    print("Deployment details: ", deployment)

    # Pause the deployment
    #cclient.pause(deployment.id)

    # Delete the deployment
    #cclient.delete(deployment.id)
