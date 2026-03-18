"""CLI commands for interactive shell and command execution in deployment pods."""

import asyncio
import sys

import click

from centml.cli.cluster import handle_exception
from centml.sdk import auth
from centml.sdk.api import get_centml_client
from centml.sdk.config import settings
from centml.sdk.shell import (
    PodNotFoundError,
    ShellError,
    build_ws_url,
    exec_session,
    get_running_pods,
    interactive_session,
)


def _resolve_pod(running_pods: list[str], pod_name: str) -> str:
    """Validate that *pod_name* exists in *running_pods*."""
    if pod_name not in running_pods:
        pods_list = ", ".join(running_pods)
        raise PodNotFoundError(f"Pod '{pod_name}' not found. Available running pods: {pods_list}")
    return pod_name


def _select_pod(running_pods, deployment_id):
    click.echo(f"Multiple running pods found for deployment {deployment_id}:")
    for i, name in enumerate(running_pods, 1):
        click.echo(f"  [{i}] {name}")

    choice = click.prompt(
        "Select a pod", type=click.IntRange(1, len(running_pods)), prompt_suffix=f" [1-{len(running_pods)}]: "
    )
    return running_pods[choice - 1]


def _connect_args(deployment_id, pod, shell_type, first_pod=False):
    """Resolve pod, build WebSocket URL, and obtain auth token."""
    with get_centml_client() as cclient:
        running_pods = get_running_pods(cclient, deployment_id)
        if not running_pods:
            raise click.ClickException(f"No running pods found for deployment {deployment_id}")

        if pod is not None:
            try:
                pod_name = _resolve_pod(running_pods, pod)
            except ShellError as exc:
                raise click.ClickException(str(exc)) from exc
        elif len(running_pods) == 1 or first_pod:
            pod_name = running_pods[0]
        else:
            pod_name = _select_pod(running_pods, deployment_id)

    ws_url = build_ws_url(settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type)
    token = auth.get_centml_token()
    return ws_url, token


@click.command(help="Open an interactive shell to a deployment pod")
@click.argument("deployment_id", type=int)
@click.option("--pod", default=None, help="Specify a pod name")
@click.option("--shell", "shell_type", default=None, type=click.Choice(["bash", "sh", "zsh"]), help="Shell type")
@click.option(
    "--first-pod", is_flag=True, default=False, help="Auto-select the first running pod (skip interactive selection)"
)
@handle_exception
def shell(deployment_id, pod, shell_type, first_pod):
    if not sys.stdin.isatty():
        raise click.ClickException("Interactive shell requires a terminal (TTY)")

    ws_url, token = _connect_args(deployment_id, pod, shell_type, first_pod)
    exit_code = asyncio.run(interactive_session(ws_url, token))
    sys.exit(exit_code)


@click.command(help="Execute a command in a deployment pod", context_settings={"ignore_unknown_options": True})
@click.argument("deployment_id", type=int)
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--pod", default=None, help="Specific pod name")
@click.option("--shell", "shell_type", default=None, type=click.Choice(["bash", "sh", "zsh"]), help="Shell type")
@click.option(
    "--first-pod", is_flag=True, default=False, help="Auto-select the first running pod (skip interactive selection)"
)
@handle_exception
def exec_cmd(deployment_id, command, pod, shell_type, first_pod):
    ws_url, token = _connect_args(deployment_id, pod, shell_type, first_pod)
    exit_code = asyncio.run(exec_session(ws_url, token, " ".join(command)))
    sys.exit(exit_code)
