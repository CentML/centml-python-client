import centml
from centml.sdk.api import get_centml_client
from centml.sdk import CreateInferenceV3DeploymentRequest
from centml.sdk.utils.config_file import load_config_file_mount


def main():
    with get_centml_client() as cclient:
        # Mounts ./vllm_config.yaml at /etc/vllm/vllm_config.yaml and lets vLLM
        # consume the whole config via --config. mount_path is the parent
        # directory; filename defaults to os.path.basename(path) so the file
        # lands at mount_path/filename. The sibling vllm_config.yaml in this
        # directory shows a realistic Llama-3.1-8B + EAGLE3 speculative-decoding
        # setup; edit it (model, dtype, tensor-parallel-size, speculative-config,
        # etc.) to match the workload before deploying.
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
            command="python -m vllm.entrypoints.openai.api_server --port 8000 --config /etc/vllm/vllm_config.yaml",
            config_file=load_config_file_mount(path="./vllm_config.yaml", mount_path="/etc/vllm"),
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
