import argparse


def main():
    parser = argparse.ArgumentParser(description="CentML command line tool")
    subparser = parser.add_subparsers(help="sub-command help", dest="mode")

    # pylint: disable=unused-variable
    server_parser = subparser.add_parser("server", help="Remote computation server")
    ccompute_parser = subparser.add_parser("cluster", help="CCluster CLI tool")
    login_parser = subparser.add_parser("login", help="Login to CentML")
    logout_parser = subparser.add_parser("logout", help="Logout from CentML")
    # pylint: enable=unused-variable

    login_parser.add_argument("token_file", help="CentML authentication token file", default=None, nargs='?')

    args = parser.parse_args()

    if args.mode == "server":
        from ..compiler import server

        server.run()
    elif args.mode == "cluster":
        print(args.mode)
    elif args.mode == "login":
        from . import login

        login.login(args.token_file)
    elif args.mode == "logout":
        from . import login

        login.logout()
    else:
        parser.print_help()
        parser.exit()
