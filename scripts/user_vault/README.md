# User Vault Scripts

Scripts for managing user vault (secrets) in CentML Platform.

## Overview

The CentML User Vault is a secure storage system for sensitive information that can be used across your deployments. This includes environment variables, API tokens, SSH keys, and certificates. These scripts allow you to view and manage your vault items from the command line.

## Prerequisites

### 1. Install the centml package

From the repository root directory:

```bash
pip install -e ./
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/CentML/centml-python-client.git@main
```

### 2. Authenticate with CentML

Login to your CentML account:

```bash
centml login
```

This will open a browser window for authentication. Once completed, your credentials will be stored locally.

## Available Scripts

### get_vault_items.py

Retrieves and displays all items stored in your CentML vault.

#### Supported Vault Types

| Type | Description | Example Use Case |
|------|-------------|------------------|
| `env_vars` | Environment variables | Database URLs, API endpoints |
| `ssh_keys` | SSH keys | Git repository access |
| `bearer_tokens` | Bearer tokens | Service authentication |
| `access_tokens` | Access tokens | HuggingFace tokens, Weights & Biases API keys |
| `certificates` | Certificates | TLS/SSL certificates |

#### Usage

Run the script from the `scripts/user_vault` directory:

```bash
cd scripts/user_vault
python get_vault_items.py [OPTIONS]
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type TYPE` | Filter results by vault type (see supported types above) | Show all types |
| `--search QUERY` | Filter items by key name (case-sensitive substring match) | No filter |
| `--show-values` | Display the actual secret values | Keys only |
| `--help` | Show help message and exit | - |

#### Examples

**List all vault items (keys only):**

```bash
python get_vault_items.py
```

**List only environment variables:**

```bash
python get_vault_items.py --type env_vars
```

**List only access tokens (e.g., HuggingFace tokens):**

```bash
python get_vault_items.py --type access_tokens
```

**Search for items containing "HF" in the key name:**

```bash
python get_vault_items.py --search HF
```

**Show all items with their values:**

```bash
python get_vault_items.py --show-values
```

**Combine multiple options:**

```bash
python get_vault_items.py --type env_vars --show-values --search DATABASE
```

#### Example Output

Without `--show-values`:

```
Found 5 vault item(s)

==================================================
Type: access_tokens (2 item(s))
==================================================
  HF_TOKEN
  WANDB_API_KEY

==================================================
Type: env_vars (3 item(s))
==================================================
  API_KEY
  DATABASE_URL
  MY_SECRET
```

With `--show-values`:

```
Found 5 vault item(s)

==================================================
Type: access_tokens (2 item(s))
==================================================
  HF_TOKEN: hf_xxxxxxxxxxxxxxxxxxxx
  WANDB_API_KEY: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

==================================================
Type: env_vars (3 item(s))
==================================================
  API_KEY: sk-xxxxxxxxxxxxxxxx
  DATABASE_URL: postgresql://user:pass@host:5432/db
  MY_SECRET: my-secret-value
```

## Troubleshooting

### Authentication Error

If you see an authentication error, try logging in again:

```bash
centml login
```

### Module Not Found

If you see `ModuleNotFoundError`, ensure you have installed the centml package:

```bash
pip install -e ./
```

### No Items Found

If the script returns "No vault items found", verify that:
1. You are logged into the correct CentML account
2. You have created vault items in the CentML web UI or via API
