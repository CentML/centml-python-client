"""CLI commands for interactive shell and command execution in deployment pods."""

import asyncio
import sys

import click

from centml.cli.cluster import handle_exception
from centml.sdk import auth
from centml.sdk.api import get_centml_client
from centml.sdk.config import settings
from centml.sdk.shell import ShellError
from centml.sdk.shell.session import build_ws_url, exec_session, interactive_session, resolve_pod


@click.command(help="Open an interactive shell to a deployment pod")
@click.argument("deployment_id", type=int)
@click.option("--pod", default=None, help="Specific pod name (auto-selects first running pod)")
@click.option("--shell", "shell_type", default=None, type=click.Choice(["bash", "sh", "zsh"]), help="Shell type")
@handle_exception
def shell(deployment_id, pod, shell_type):
    if not sys.stdin.isatty():
        raise click.ClickException("Interactive shell requires a terminal (TTY)")

    with get_centml_client() as cclient:
        try:
            pod_name, warning = resolve_pod(cclient, deployment_id, pod)
        except ShellError as exc:
            raise click.ClickException(str(exc)) from exc
    if warning:
        click.echo(f"{warning} Use --pod to specify a different pod.", err=True)

    ws_url = build_ws_url(settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type)
    token = auth.get_centml_token()
    exit_code = asyncio.run(interactive_session(ws_url, token))
    sys.exit(exit_code)


@click.command(help="Execute a command in a deployment pod", context_settings={"ignore_unknown_options": True})
@click.argument("deployment_id", type=int)
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--pod", default=None, help="Specific pod name")
@click.option("--shell", "shell_type", default=None, type=click.Choice(["bash", "sh", "zsh"]), help="Shell type")
@handle_exception
def exec_cmd(deployment_id, command, pod, shell_type):
    with get_centml_client() as cclient:
        try:
            pod_name, warning = resolve_pod(cclient, deployment_id, pod)
        except ShellError as exc:
            raise click.ClickException(str(exc)) from exc
    if warning:
        click.echo(f"{warning} Use --pod to specify a different pod.", err=True)

    ws_url = build_ws_url(settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type)
    token = auth.get_centml_token()
    cmd_str = " ".join(command)
    exit_code = asyncio.run(exec_session(ws_url, token, cmd_str))
    sys.exit(exit_code)
