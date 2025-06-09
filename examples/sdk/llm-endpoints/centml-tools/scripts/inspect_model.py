import sys
import json
import os
from centml.sdk.api import get_centml_client
from centml.sdk import CServeV2Recipe


def print_hardware_details(cclient, instance_id):
    hardware = next((h for h in cclient.get_hardware_instances() if h.id == instance_id), None)
    if hardware:
        print(f"  🧠 Hardware ID {hardware.id}:")
        print(
            f"    name='{hardware.name}' gpu_type='{hardware.gpu_type}' num_gpu={hardware.num_gpu} "
            f"cpu={hardware.cpu} memory={hardware.memory} cost_per_hr={hardware.cost_per_hr} "
            f"cluster_id={hardware.cluster_id} provider='{hardware.provider}' "
            f"num_accelerators={hardware.num_accelerators} accelerator_memory={hardware.accelerator_memory}"
        )
    else:
        print(f"  ⚠️ Hardware ID {instance_id} not found.")


def print_all_config_variants(recipe_variant, cclient):
    for label in dir(recipe_variant):
        if label.startswith("_"):
            continue
        config = getattr(recipe_variant, label, None)
        if config and hasattr(config, "recipe"):
            print(f"\n🔧 Prebuilt Configuration: {label}")
            print(f"  Model: {config.recipe.model}")
            print(f"  Hardware Instance ID: {config.hardware_instance_id}")
            print_hardware_details(cclient, config.hardware_instance_id)
            print("  Recipe:")
            try:
                print(config.recipe.model_dump())
            except AttributeError:
                print(config.recipe.dict())


def print_all_hardware(cclient):
    print("\n📋 All Available Hardware Instances:")
    for h in cclient.get_hardware_instances():
        print(
            f"id={h.id} name='{h.name}' gpu_type='{h.gpu_type}' num_gpu={h.num_gpu} "
            f"cpu={h.cpu} memory={h.memory} cost_per_hr={h.cost_per_hr} "
            f"cluster_id={h.cluster_id} provider='{h.provider}' "
            f"num_accelerators={h.num_accelerators} accelerator_memory={h.accelerator_memory}"
        )


def main():
    model_name = None
    config_path = None

    for arg in sys.argv[1:]:
        if arg.startswith("CONFIG="):
            config_path = arg.split("=", 1)[1]
        elif arg.endswith(".json") and os.path.isfile(arg):
            config_path = arg
        elif os.path.isfile(arg):
            config_path = arg
        else:
            model_name = arg  # fallback to direct model name

    if config_path:
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                model_name = config.get("model")
        except Exception as e:
            print(f"❌ Failed to read config: {e}")
            sys.exit(1)

    if not model_name:
        print("❌ Usage: python3 inspect_model.py <model_name> or CONFIG=<config.json>")
        sys.exit(1)

    with get_centml_client() as cclient:
        print(f"\n🔍 Inspecting model: {model_name}")
        recipes = cclient.get_cserve_recipe(model=model_name)

        for variant in recipes:
            print_all_config_variants(variant, cclient)

        print_all_hardware(cclient)




if __name__ == "__main__":
    main()
