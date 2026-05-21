import os
from typing import Optional

from platform_api_python_client import ConfigFileMount


# Load a file off disk into a ConfigFileMount. Field-level validation
# (size cap, filename charset, mount_path rules) is intentionally left
# to the API so SDK doesn't drift when server limits change.
def load_config_file_mount(path: str, mount_path: str, filename: Optional[str] = None) -> ConfigFileMount:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return ConfigFileMount(filename=filename or os.path.basename(path), mount_path=mount_path, content=content)
