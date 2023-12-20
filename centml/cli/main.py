import argparse

from ..compiler import server


def main():
    parser = argparse.ArgumentParser(description="CentML command line tool")
    subparser = parser.add_subparsers(help="sub-command help", dest="mode")

    server_parser = subparser.add_parser("server", help="Remote computation server")

    ccompute_parser = subparser.add_parser("cluster", help="CCompute CLI tool")

    args = parser.parse_args()

    if args.mode == "server":
        server.run()
    elif args.mode == "cluster":
        print(args.mode)
    else:
        parser.print_help()
        parser.exit()
