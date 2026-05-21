import os
from typing import Optional

from platform_api_python_client import ConfigFileMount


# Load a file off disk into a ConfigFileMount. `mount_path` is the parent
# directory inside the container; the file lands at `mount_path/filename`.
# Field-level validation (size cap, filename charset, mount_path rules) is
# left to the API so SDK doesn't drift when server limits change.
def load_config_file_mount(path: str, mount_path: str, filename: Optional[str] = None) -> ConfigFileMount:
    # newline="" disables universal-newline translation so CRLF/CR line
    # endings reach the server byte-faithful instead of being normalized to \n.
    with open(path, "r", encoding="utf-8", newline="") as f:
        content = f.read()
    return ConfigFileMount(filename=filename or os.path.basename(path), mount_path=mount_path, content=content)
