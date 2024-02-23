import argparse
import click

from .login import login, logout
from .cluster import ls, deploy, delete, status

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
ccluster.add_command(deploy)
ccluster.add_command(delete)
ccluster.add_command(status)


cli.add_command(ccluster, name="cluster")

def main():
    parser = argparse.ArgumentParser(description="CentML command line tool")
    subparser = parser.add_subparsers(help="sub-command help", dest="mode")

    # pylint: disable=unused-variable
    server_parser = subparser.add_parser("server", help="Remote computation server")
    logout_parser = subparser.add_parser("logout", help="Logout from CentML")
    # pylint: enable=unused-variable

    login_parser = subparser.add_parser("login", help="Login to CentML")
    login_parser.add_argument("token_file", help="CentML authentication token file", default=None, nargs='?')

    cluster_parser = subparser.add_parser("cluster", help="CCluster CLI tool")
    cluster_parser.add_argument("cmd", help="CCluster command", choices=['ls', 'deploy', 'delete', 'status'])
    cluster_parser.add_argument("--id", help="Deployment id")
    cluster_parser.add_argument("--name", help="Deployment name")
    cluster_parser.add_argument("--image", help="Container image url")
    cluster_parser.add_argument("--port", help="Port to expose")

    args = parser.parse_args()

    if args.mode == "server":
        from ..compiler import server

        server.run()
    elif args.mode == "cluster":
        from . import cluster

        cluster.run(args)
    elif args.mode == "login":
        from . import login

        login.login(args.token_file)
    elif args.mode == "logout":
        from . import login

        login.logout()
    else:
        parser.print_help()
        parser.exit()
