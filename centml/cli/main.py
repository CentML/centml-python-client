import argparse
import click

from .login import login, logout
from .cluster import ls, get, deploy, delete, status

@click.group()
def cli():
    pass


cli.add_command(login)
cli.add_command(logout)


@cli.command(help="Start remote compilation server")
def server():
    from .. import compiler

    compiler.server.run()


@click.group(help="CentML cluster CLI tool")
def ccluster():
    pass


ccluster.add_command(ls)
ccluster.add_command(get)
ccluster.add_command(deploy)
ccluster.add_command(delete)
ccluster.add_command(status)


cli.add_command(ccluster, name="cluster")
