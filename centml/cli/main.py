import argparse

from ..compiler import server


def main():
    parser = argparse.ArgumentParser(description="CentML command line tool")
    parser.add_argument("mode", choices=["compile-server"], type=str)

    args = parser.parse_args()

    if args.mode == "compile-server":
        server.run()
    else:
        raise Exception("Invalid mode")
