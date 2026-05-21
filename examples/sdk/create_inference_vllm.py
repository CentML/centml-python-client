import centml
from centml.sdk.api import get_centml_client
from centml.sdk import CreateInferenceV3DeploymentRequest
from centml.sdk.utils.config_file import load_config_file_mount


def main():
    with get_centml_client() as cclient:
        # Mounts ./chat_template.jinja at /etc/vllm/chat_template.jinja and
        # tells vLLM to use it via --chat-template. mount_path is the parent
        # directory; filename defaults to os.path.basename(path).
        request = CreateInferenceV3DeploymentRequest(
            name="vllm-llama",
            cluster_id=1000,
            hardware_instance_id=1001,  # GPU instance
            image_url="vllm/vllm-openai:latest",
            port=8000,
            min_replicas=1,
            max_replicas=1,
            initial_replicas=1,
            max_surge=1,
            max_unavailable=0,
            healthcheck="/health",
            concurrency=10,
            env_vars={"HF_TOKEN": "<your-hf-token>"},
            command=(
                "python -m vllm.entrypoints.openai.api_server "
                "--model meta-llama/Llama-3.2-3B-Instruct "
                "--port 8000 "
                "--chat-template /etc/vllm/chat_template.jinja"
            ),
            config_file=load_config_file_mount(path="./chat_template.jinja", mount_path="/etc/vllm"),
        )
        response = cclient.create_inference(request)
        print("Create deployment response: ", response)

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
