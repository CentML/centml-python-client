# 🚀 CentML Endpoint Creator

This script creates a CentML CServe V2 deployment for a given model, using either the fastest available hardware or a default hardware configuration. This is meant to be an example to help users work with the CentML SDK. 

---

## 📄 Script: `create_endpoint.py`

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

📬 Questions?

Reach out to the CentML team or maintainers if you encounter unexpected recipe/hardware mismatches.
