import json
import os
import sys
from centml.sdk.api import get_centml_client
from centml.sdk import CreateCServeV2DeploymentRequest, CServeV2Recipe
from platform_api_python_client.exceptions import BadRequestException

def list_prebuilt_recipes(model_name, cclient):
    print(f"\n🔍 Looking up prebuilt recipes for model: {model_name}")
    recipes = cclient.get_cserve_recipe(model=model_name)
    labeled = []

    for variant in recipes:
        for label in dir(variant):
            if not label.startswith("_"):
                config = getattr(variant, label, None)
                if config and hasattr(config, "recipe"):
                    print(f"{len(labeled)}. {label} - hardware ID {config.hardware_instance_id}")
                    labeled.append(config)
    return labeled

def write_template_file(filename, config_data):
    with open(filename, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"📁 Template written to {filename}")

def validate_hardware_and_cluster(config, cclient):
    hardware_instances = cclient.get_hardware_instances()
    for h in hardware_instances:
        if h.id == config["hardware_instance_id"] and h.cluster_id == config["cluster_id"]:
            return True
    return False

def load_or_create_config(filename, cclient):
    if not os.path.exists(filename):
        print(f"❌ Config file {filename} not found.")
        template = {
            "model": "",
            "deployment_name": "",
            "hardware_instance_id": 0,
            "cluster_id": 0,
            "min_scale": 1,
            "max_scale": 1,
            "recipe": {
                "model": "",
                "is_embedding_model": False,
                "additional_properties": {
                    "revision": None,
                    "seed": 0,
                    "dtype": "auto",
                    "tokenizer": "",
                    "block_size": 32,
                    "swap_space": 0,
                    "download_dir": None,
                    "gpu_mem_util": 0.9,
                    "max_num_seqs": 1024,
                    "quantization": None,
                    "max_model_len": None,
                    "tokenizer_mode": "auto",
                    "use_flashinfer": True,
                    "eager_execution": False,
                    "engine_managers": ["localhost:6061"],
                    "dist_init_method": None,
                    "num_scheduler_steps": 1,
                    "tensor_parallel_size": 1,
                    "environment_variables": {
                        "apply": {
                            "NCCL_SHM_DISABLE": 1
                        }
                    },
                    "max_seq_len_to_capture": None,
                    "pipeline_parallel_size": 1,
                    "distributed_executor_backend": "uni"
                }
            }
        }
        write_template_file(filename, template)
        use_prebuilt = input("Would you like to use a prebuilt recipe instead? (y/n): ").strip().lower()
        if use_prebuilt == "y":
            model = input("Enter the model name: ").strip()
            options = list_prebuilt_recipes(model, cclient)
            if not options:
                print("⚠️ No prebuilt recipes found. Please edit the template and try again.")
                sys.exit(1)
            index = int(input("Select a recipe by number: ").strip())
            selected = options[index]
            config_data = {
                "model": selected.recipe.model,
                "deployment_name": input("Enter a deployment name (max 20 chars): ").strip(),
                "hardware_instance_id": selected.hardware_instance_id,
                "cluster_id": cclient.get_cluster_id(selected.hardware_instance_id),
                "min_scale": 0,
                "max_scale": 1,
                "recipe": json.loads(selected.recipe.model_dump_json())
            }
            write_template_file(filename, config_data)
            return config_data
        else:
            print("📝 Please edit the generated template with the correct values and re-run.")
            sys.exit(1)

    with open(filename, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            print("❌ Invalid JSON. Please fix the file and try again.")
            sys.exit(1)

    required_fields = ("model", "deployment_name", "cluster_id", "hardware_instance_id", "min_scale", "max_scale")
    if not all(k in config and config[k] is not None for k in required_fields):
        print("⚠️ Missing required fields in config. Please edit the file and try again.")
        sys.exit(1)

    if not isinstance(config["min_scale"], int) or not isinstance(config["max_scale"], int):
        print("❌ min_scale and max_scale must be integers.")
        sys.exit(1)

    if not validate_hardware_and_cluster(config, cclient):
        print("❌ Invalid hardware_instance_id and cluster_id combination.")
        print("\n📋 Available Hardware Instances:")
        for h in cclient.get_hardware_instances():
            print(
                f"id={h.id} name='{h.name}' gpu_type='{h.gpu_type}' num_gpu={h.num_gpu} "
                f"cpu={h.cpu} memory={h.memory} cost_per_hr={h.cost_per_hr} "
                f"cluster_id={h.cluster_id} provider='{h.provider}' "
                f"num_accelerators={h.num_accelerators} accelerator_memory={h.accelerator_memory}"
            )

        print("\n🤖 Suggested valid options from prebuilt recipes:")
        try:
            prebuilt_options = cclient.get_cserve_recipe(model=config["model"])
            for variant in prebuilt_options:
                for label in dir(variant):
                    if not label.startswith("_"):
                        option = getattr(variant, label, None)
                        if option and hasattr(option, "hardware_instance_id"):
                            cluster_id = cclient.get_cluster_id(option.hardware_instance_id)
                            print(f" - hardware_instance_id: {option.hardware_instance_id}, cluster_id: {cluster_id} ({label})")
        except Exception as e:
            print("⚠️ Could not retrieve prebuilt recipes for suggestions.")

        sys.exit(1)

    return config

def deploy_model(config, cclient):
    print(f"\n🚀 Deploying model: {config['model']} as '{config['deployment_name']}'")

    request = CreateCServeV2DeploymentRequest(
        name=config["deployment_name"],
        cluster_id=config["cluster_id"],
        hardware_instance_id=config["hardware_instance_id"],
        recipe=CServeV2Recipe(**config["recipe"]),
        min_scale=config.get("min_scale", 1),
        max_scale=config.get("max_scale", 1),
        env_vars={}
    )

    print("\n📋 Checking all existing deployments (all types)...")
    try:
        deployments = cclient._api.get_deployments_deployments_get().results
    except Exception as e:
        print(f"❌ Failed to fetch deployments: {e}")
        sys.exit(1)

    print("\n📋 Raw deployments returned:")
    for d in deployments:
        print(f"  - name: '{getattr(d, 'name', '')}' (id={getattr(d, 'id', '')}, type={getattr(d, 'type', '')})")

    target = config["deployment_name"].strip().lower()
    matched = next(
        (d for d in deployments if getattr(d, "name", "").strip().lower() == target),
        None
    )

    if matched:
        print(f"\n🔄 Deployment '{config['deployment_name']}' exists (ID: {matched.id}). Proceeding with update...")
        try:
            response = cclient.update_cserve(matched.id, request)
            print("✅ Deployment updated successfully.")
        except Exception as e:
            print(f"❌ Failed to update deployment: {e}")
            raise
    else:
        print(f"\n🆕 Deployment '{config['deployment_name']}' not found. Proceeding with creation...")
        try:
            response = cclient.create_cserve(request)
            print("✅ Deployment created successfully.")
        except Exception as e:
            print(f"❌ Failed to create deployment: {e}")
            raise

    print("📦 Deployment ID:", response.id)
    print("🔧 You can now monitor or manage your deployment using the CentML dashboard or CLI.")

def main():
    if len(sys.argv) < 2:
        print("📁 No config file provided. Defaulting to 'config.json'...")
        config_file = "config.json"
    else:
        config_file = sys.argv[1]

    with get_centml_client() as cclient:
        config = load_or_create_config(config_file, cclient)
        deploy_model(config, cclient)

if __name__ == "__main__":
    main()
