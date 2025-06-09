# 🧠 CentML Deployment Tools

This repository provides a simple Makefile interface to deploy, inspect, and delete model deployments using the CentML SDK.

## 📦 Prerequisites

- Python 3.11 (or update the `PYTHON` path in the Makefile)
- The CentML Python SDK installed:  
  ```bash
  pip install centml
## 🛠️ Usage
Run any of the following commands from the project root:
### 🚀 Deploy a Model
`make deploy CONFIG=test_config.json` 
Deploys a model based on the configuration in test_config.json.
### 🔍 Inspect a Model
`make inspect CONFIG="model_name"
Displays model about the deployment with name model_name.
Or, if you have the config file:
`make inspect CONFIG=test_config.json`
### Register a Deployment
WIP 
