import click

from centml.cli.login import login, logout
from centml.cli.cluster import ls, get, delete, pause, resume


@click.group()
def cli():
    pass


cli.add_command(login)
cli.add_command(logout)


@cli.command(help="Start remote compilation server")
def server():
    from centml.compiler.server import run

    run()


@click.group(help="CentML cluster CLI tool")
def ccluster():
    pass


ccluster.add_command(ls)
ccluster.add_command(get)
ccluster.add_command(delete)
ccluster.add_command(pause)
ccluster.add_command(resume)


cli.add_command(ccluster, name="cluster")
