"""CLI commands for interactive shell and command execution in deployment pods."""

import asyncio
import json
import re
import shutil
import signal
import sys
import urllib.parse

import click

from centml.sdk import PodStatus
from centml.sdk import auth
from centml.sdk.api import get_centml_client
from centml.sdk.config import settings
from centml.cli.cluster import handle_exception

# Lazy-import to keep module loadable without websockets installed at import time,
# and to allow tests to patch the module attribute easily.
import websockets

# These are only available on Unix; guarded at command level via isatty check.
import termios
import tty


def _build_ws_url(api_url, deployment_id, pod_name, shell=None):
    """Build the WebSocket URL for a terminal connection."""
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = parsed._replace(scheme=ws_scheme).geturl()
    url = f"{ws_base}/deployments/{deployment_id}/terminal?pod={urllib.parse.quote(pod_name)}"
    if shell:
        url += f"&shell={urllib.parse.quote(shell)}"
    return url


def _resolve_pod(cclient, deployment_id, pod_name=None):
    """Resolve which pod to connect to.

    Args:
        cclient: CentMLClient instance.
        deployment_id: The deployment ID.
        pod_name: Optional specific pod name to target.

    Returns:
        The pod name string to connect to.

    Raises:
        click.ClickException: If no running pods or specified pod not found.
    """
    status = cclient.get_status_v3(deployment_id)
    running_pods = []
    for revision in status.revision_pod_details_list or []:
        for pod in revision.pod_details_list or []:
            if pod.status == PodStatus.RUNNING and pod.name:
                running_pods.append(pod.name)

    if not running_pods:
        raise click.ClickException(
            f"No running pods found for deployment {deployment_id}"
        )

    if pod_name is not None:
        if pod_name not in running_pods:
            pods_list = ", ".join(running_pods)
            raise click.ClickException(
                f"Pod '{pod_name}' not found. Available running pods: {pods_list}"
            )
        return pod_name

    if len(running_pods) > 1:
        click.echo(
            f"Multiple running pods found, connecting to {running_pods[0]}. "
            f"Use --pod to specify a different pod.",
            err=True,
        )
    return running_pods[0]


async def _forward_io(ws):
    """Bidirectional forwarding between local stdin/stdout and WebSocket.

    Returns the remote exit code.
    """
    loop = asyncio.get_running_loop()
    exit_code = 0
    stdin_fd = sys.stdin.fileno()

    stdin_closed = asyncio.Event()

    async def _read_ws():
        nonlocal exit_code
        async for raw_msg in ws:
            msg = json.loads(raw_msg)
            if msg.get("data"):
                sys.stdout.buffer.write(msg["data"].encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            elif msg.get("error"):
                sys.stderr.write(f"Error: {msg['error']}\n")
                sys.stderr.flush()
            if "Code" in msg:
                exit_code = msg["Code"]
                return

    async def _read_stdin():
        read_queue = asyncio.Queue()

        def _on_stdin_ready():
            data = sys.stdin.buffer.read1(4096)
            if data:
                read_queue.put_nowait(data)
            else:
                stdin_closed.set()

        loop.add_reader(stdin_fd, _on_stdin_ready)
        try:
            while not stdin_closed.is_set():
                try:
                    data = await asyncio.wait_for(read_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                rows, cols = shutil.get_terminal_size()
                await ws.send(
                    json.dumps(
                        {
                            "operation": "stdin",
                            "data": data.decode("utf-8", errors="replace"),
                            "rows": rows,
                            "cols": cols,
                        }
                    )
                )
        finally:
            loop.remove_reader(stdin_fd)

    tasks = [
        asyncio.create_task(_read_ws()),
        asyncio.create_task(_read_stdin()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        if t.exception() is not None:
            raise t.exception()
    return exit_code


async def _interactive_session(ws_url, token):
    """Run an interactive terminal session over WebSocket.

    Enters raw mode, forwards I/O bidirectionally, and restores terminal on exit.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rows, cols = shutil.get_terminal_size()

        headers = {"Authorization": f"Bearer {token}"}
        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            await ws.send(
                json.dumps(
                    {
                        "operation": "resize",
                        "rows": rows,
                        "cols": cols,
                    }
                )
            )

            loop = asyncio.get_running_loop()

            def _on_resize():
                r, c = shutil.get_terminal_size()
                asyncio.ensure_future(
                    ws.send(
                        json.dumps(
                            {
                                "operation": "resize",
                                "rows": r,
                                "cols": c,
                            }
                        )
                    )
                )

            loop.add_signal_handler(signal.SIGWINCH, _on_resize)

            try:
                exit_code = await _forward_io(ws)
            finally:
                loop.remove_signal_handler(signal.SIGWINCH)

            return exit_code
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?\x1b\\")

_BEGIN_MARKER = "__CENTML_BEGIN_5f3a__"
_END_MARKER = "__CENTML_END_5f3a__"

# printf octal \137 = underscore. The decoded output matches _BEGIN/_END_MARKER,
# but the literal command text does NOT, so shell echo won't trigger false matches.
_PRINTF_BEGIN = r"\137\137CENTML_BEGIN_5f3a\137\137"
_PRINTF_END = r"\137\137CENTML_END_5f3a\137\137"


def _strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


async def _exec_session(ws_url, token, command):
    """Execute a command in a pod and return its exit code.

    Does not enter raw mode -- output is pipe-friendly.
    Suppresses shell echo and uses markers to capture only command output.
    """
    rows, cols = shutil.get_terminal_size(fallback=(80, 24))
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "operation": "resize",
                    "rows": rows,
                    "cols": cols,
                }
            )
        )

        # Suppress echo/bracketed-paste, emit begin marker, run command,
        # emit end marker with exit code, then exit.
        # Markers use printf octal escapes so the literal marker string
        # doesn't appear in the command echo.
        wrapped = (
            f"stty -echo 2>/dev/null; printf '\\033[?2004l';"
            f" printf '{_PRINTF_BEGIN}\\n';"
            f" {command};"
            f" __ec=$?;"
            f" printf '\\n{_PRINTF_END}:%d\\n' \"$__ec\";"
            f" exit $__ec\n"
        )

        await ws.send(
            json.dumps(
                {
                    "operation": "stdin",
                    "data": wrapped,
                    "rows": rows,
                    "cols": cols,
                }
            )
        )

        exit_code = 0
        buffer = ""
        is_capturing = False
        async for raw_msg in ws:
            msg = json.loads(raw_msg)
            if msg.get("data"):
                buffer += msg["data"]
                # Process complete lines from buffer
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    clean = _strip_ansi(line).rstrip("\r")
                    if _BEGIN_MARKER in clean:
                        is_capturing = True
                        continue
                    if _END_MARKER in clean:
                        # Parse exit code from marker line
                        parts = clean.split(_END_MARKER + ":")
                        if len(parts) > 1:
                            try:
                                exit_code = int(parts[1].strip())
                            except ValueError:
                                pass
                        is_capturing = False
                        continue
                    if is_capturing:
                        sys.stdout.write(line + "\n")
                        sys.stdout.flush()
            elif msg.get("error"):
                sys.stderr.write(f"Error: {msg['error']}\n")
                return 1
            if "Code" in msg:
                exit_code = msg["Code"]
                break
        return exit_code


@click.command(help="Open an interactive shell to a deployment pod")
@click.argument("deployment_id", type=int)
@click.option(
    "--pod", default=None, help="Specific pod name (auto-selects first running pod)"
)
@click.option(
    "--shell",
    "shell_type",
    default=None,
    type=click.Choice(["bash", "sh", "zsh"]),
    help="Shell type",
)
@handle_exception
def shell(deployment_id, pod, shell_type):
    if not sys.stdin.isatty():
        raise click.ClickException("Interactive shell requires a terminal (TTY)")

    with get_centml_client() as cclient:
        pod_name = _resolve_pod(cclient, deployment_id, pod)

    ws_url = _build_ws_url(
        settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type
    )
    token = auth.get_centml_token()
    exit_code = asyncio.run(_interactive_session(ws_url, token))
    sys.exit(exit_code)


@click.command(
    help="Execute a command in a deployment pod",
    context_settings=dict(ignore_unknown_options=True),
)
@click.argument("deployment_id", type=int)
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--pod", default=None, help="Specific pod name")
@click.option(
    "--shell",
    "shell_type",
    default=None,
    type=click.Choice(["bash", "sh", "zsh"]),
    help="Shell type",
)
@handle_exception
def exec_cmd(deployment_id, command, pod, shell_type):
    with get_centml_client() as cclient:
        pod_name = _resolve_pod(cclient, deployment_id, pod)

    ws_url = _build_ws_url(
        settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type
    )
    token = auth.get_centml_token()
    cmd_str = " ".join(command)
    exit_code = asyncio.run(_exec_session(ws_url, token, cmd_str))
    sys.exit(exit_code)
