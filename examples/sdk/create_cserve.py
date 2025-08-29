import centml
from centml.sdk.api import get_centml_client
from centml.sdk import DeploymentType, CreateCServeV3DeploymentRequest, CServeV2Recipe


def get_fastest_cserve_config(cclient, name, model):
    fastest = cclient.get_cserve_recipe(model=model)[0].fastest

    return CreateCServeV3DeploymentRequest(
        name=name,
        cluster_id=cclient.get_cluster_id(fastest.hardware_instance_id),
        hardware_instance_id=fastest.hardware_instance_id,
        recipe=fastest.recipe,
        min_replicas=1,
        max_replicas=1,
        env_vars={},
    )


def get_default_cserve_config(cclient, name, model):
    default_recipe = CServeV2Recipe(model=model)

    hardware_instance = cclient.get_hardware_instances(cluster_id=1001)[0]

    return CreateCServeV3DeploymentRequest(
        name=name,
        cluster_id=hardware_instance.cluster_id,
        hardware_instance_id=hardware_instance.id,
        recipe=default_recipe,
        min_replicas=1,
        max_replicas=1,
        env_vars={},
    )


def main():
    with get_centml_client() as cclient:
        ### Get the configurations for the Qwen model
        qwen_config = get_fastest_cserve_config(
            cclient, name="qwen-fastest", model="Qwen/Qwen2-VL-7B-Instruct"
        )
        # qwen_config = get_default_cserve_config(cclient, name="qwen-default", model="Qwen/Qwen2-VL-7B-Instruct")

        ### Modify the recipe if necessary
        qwen_config.recipe.additional_properties["max_num_seqs"] = 512

        # Create CServeV3 deployment
        response = cclient.create_cserve(qwen_config)
        print("Create deployment response: ", response)

        ### Get deployment details
        deployment = cclient.get_cserve_v3(response.id)
        print("Deployment details: ", deployment)

        """
        ### Pause the deployment
        cclient.pause(deployment.id)

        ### Delete the deployment
        cclient.delete(deployment.id)
        """


if __name__ == "__main__":
    main()
