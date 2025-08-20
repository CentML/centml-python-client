# 🧠 CentML Deployment Toolkit

This directory contains Python utilities to manage CentML model deployments, query available hardware, and clean up resources.

---

## 📦 Tools Overview

| Script                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `deploy_model.py`       | Creates or updates a model deployment from a config JSON                   |
| `delete_deployment.py`  | Deletes a deployment by name using a config JSON                           |
| `inspect_model.py`      | Lists all available hardware and deployment recipes for a given model      |
| `create_endpoint.py`    | Fetches the fastest  CServe recipe for the specified model and deploys it  |


## 🚀 Deployment Script

### 📄 File: `deploy_model.py`

This script deploys a model to CentML using the CServe V2 API. It will:

1. Load a JSON configuration file.
2. Validate cluster and hardware instance availability.
3. Check if the deployment already exists:
   - If it does, it updates it.
   - If it doesn't, it creates a new deployment.

### ✅ JSON Config Example:

```json
{
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "deployment_name": "sample",
  "hardware_instance_id": 1086,
  "cluster_id": 1001,
  "min_scale": 1,
  "max_scale": 1,
  "recipe": {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "is_embedding_model": false,
    "additional_properties": {
      "tokenizer": "meta-llama/Llama-3.2-3B-Instruct",
      "dtype": "auto",
      "tensor_parallel_size": 1
    }
  }
}


### Usage

`python3 deploy_model.py <config_file.json>`

## Deletion Script

### 📄 File: `delete_deployment.py` 
This script deletes an existing deployment if one with the provided deployment_name exists.

It does not raise an error if the deployment isn't found — it exits cleanly. Uses the same JSON config as the deploy_model.


### Usage 
`python3 delete_deployment.py <config_file.json>`


## 🔍 Inspect Model Script
### 📄 File: `inspect_model.py`
Inspects available deployment recipes and hardware for a specific model.

### Usage 
`python3 inspect_model.py meta-llama/Llama-3.2-3B-Instruct`
The script will:

    List all recipe variants (e.g., fastest, cheapest)

    Print detailed hardware specs for each variant

    Display all available hardware options


## 📄 File: `create_endpoint.py`

### 🔧 What it does

- Fetches the fastest available CServe recipe for the specified model
- Builds a deployment request with appropriate cluster and hardware info
- Optionally modifies recipe properties (e.g. `max_num_seqs`)
- Submits the deployment via the CentML SDK
- Prints the deployment response and metadata

---

## 🧰 Requirements

- Python 3.8+
- [CentML Python SDK](https://pypi.org/project/centml/)

Install:

```bash
pip install centml
```
## Default behavior:

    Uses the fastest recipe from get_cserve_recipe(...)

    Falls back to hardcoded cluster ID 1001 in get_default_cserve_config(...) if needed

You can adjust the model and deployment name here:
```python
qwen_config = get_fastest_cserve_config(
    cclient,
    name="qwen-fastest",
    model="Qwen/Qwen2-VL-7B-Instruct"
)
```
Or use the default config instead:

```python
qwen_config = get_default_cserve_config(
    cclient,
    name="qwen-default",
    model="Qwen/Qwen2-VL-7B-Instruct"
)
```
🧪 Running the Script

`python3 create_endpoint.py`





## 🧰 Prerequisites

* Python 3.8+
* CentML Python SDK
* Valid CentML credentials (e.g., via environment or local config)

### Install Dependencies
`pip install centml`



📬 Questions?

Reach out to the CentML team or maintainers if you encounter unexpected recipe/hardware mismatches.
