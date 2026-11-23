import click

from centml.cli.login import login, logout
from centml.cli.cluster import ls, get, delete, pause, resume, capacity, register
from centml.cli.shell import shell, exec_cmd

# NVIDIA green (#76B900)
_NVIDIA_GREEN = (118, 185, 0)

_BANNER = r"""
   ______________           __           
  / ____/ ____/ /_  _______/ /____  _____
 / /   / /   / / / / / ___/ __/ _ \/ ___/
/ /___/ /___/ / /_/ (__  ) /_/  __/ /    
\____/\____/_/\__,_/____/\__/\___/_/     
"""

_VERSION_MESSAGE = click.style(_BANNER, fg=_NVIDIA_GREEN) + """
    🚀 Welcome to %(prog)s v%(version)s 🚀

     ✨ AI Deployment Made Simple ✨
📚 Documentation: https://docs.nvidia.com/ccluster
💬 Need help? Reach out on GitHub: https://github.com/NVIDIA/ccluster
"""


@click.group()
# this is the version and prog name set in setup.py
@click.version_option(prog_name="CCluster CLI", message=_VERSION_MESSAGE)
def cli():
    pass


cli.add_command(login)
cli.add_command(logout)


@click.group(help="CentML cluster CLI tool")
def ccluster():
    pass


ccluster.add_command(ls)
ccluster.add_command(get)
ccluster.add_command(delete)
ccluster.add_command(pause)
ccluster.add_command(resume)
ccluster.add_command(capacity)
ccluster.add_command(register)
ccluster.add_command(shell)
ccluster.add_command(exec_cmd, name="exec")


cli.add_command(ccluster, name="cluster")
