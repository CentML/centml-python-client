import argparse

from . import login
from ..compiler import server


def main():
    parser = argparse.ArgumentParser(description="CentML command line tool")
    subparser = parser.add_subparsers(help="sub-command help", dest="mode")

    server_parser = subparser.add_parser("server", help="Remote computation server")
    ccompute_parser = subparser.add_parser("cluster", help="CCluster CLI tool")
    login_parser = subparser.add_parser("login", help="Login to CentML")
    logout_parser = subparser.add_parser("logout", help="Logout from CentML")

    login_parser.add_argument("token_file", help="CentML authentication token file", default=None, nargs='?')

    args = parser.parse_args()

    if args.mode == "server":
        server.run()
    elif args.mode == "cluster":
        print(args.mode)
    elif args.mode == "login":
        login.login(args.token_file)
    elif args.mode == "logout":
        login.logout()
    else:
        parser.print_help()
        parser.exit()
