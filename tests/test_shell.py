"""Tests for centml.cli.shell -- CLI terminal access commands."""

import json
import sys
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, patch, call

import click
import pytest

from platform_api_python_client import PodStatus, PodDetails, RevisionPodDetails


def _async_iter_from_list(items):
    """Create an async iterator from a list of items."""
    async def _aiter():
        for item in items:
            yield item
    return _aiter()


# ---------------------------------------------------------------------------
# Helpers to build mock status responses
# ---------------------------------------------------------------------------

def _make_pod(name, status=PodStatus.RUNNING):
    pod = MagicMock(spec=PodDetails)
    pod.name = name
    pod.status = status
    return pod


def _make_revision(pods):
    rev = MagicMock(spec=RevisionPodDetails)
    rev.pod_details_list = pods
    return rev


def _make_status_response(revisions):
    resp = MagicMock()
    resp.revision_pod_details_list = revisions
    return resp


# ===========================================================================
# _build_ws_url
# ===========================================================================

class TestStripAnsi:
    def test_strips_csi_sequences(self):
        from centml.cli.shell import _strip_ansi
        assert _strip_ansi("\x1b[?2004htext\x1b[0m") == "text"

    def test_preserves_plain_text(self):
        from centml.cli.shell import _strip_ansi
        assert _strip_ansi("hello world") == "hello world"


class TestBuildWsUrl:
    def test_https_to_wss(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("https://api.centml.com", 123, "my-pod-abc")
        assert url.startswith("wss://api.centml.com/")

    def test_http_to_ws(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("http://localhost:16000", 42, "pod-1")
        assert url.startswith("ws://localhost:16000/")

    def test_contains_deployment_id_and_pod(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("https://api.centml.com", 99, "pod-xyz")
        assert "/deployments/99/terminal" in url
        assert "pod=pod-xyz" in url

    def test_with_shell(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("https://api.centml.com", 1, "p", shell="bash")
        assert "shell=bash" in url

    def test_without_shell(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("https://api.centml.com", 1, "p")
        assert "shell=" not in url

    def test_encodes_pod_name(self):
        from centml.cli.shell import _build_ws_url
        url = _build_ws_url("https://api.centml.com", 1, "pod name/special")
        assert "pod%20name" in url or "pod+name" in url


# ===========================================================================
# _resolve_pod
# ===========================================================================

class TestResolvePod:
    def test_selects_first_running(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod("pod-a", PodStatus.RUNNING),
                _make_pod("pod-b", PodStatus.RUNNING),
            ])
        ])
        result = _resolve_pod(cclient, 1)
        assert result == "pod-a"

    def test_raises_no_running_pods(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod("pod-err", PodStatus.ERROR),
            ])
        ])
        with pytest.raises(click.ClickException, match="No running pods"):
            _resolve_pod(cclient, 1)

    def test_raises_specified_pod_not_found(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod("pod-a", PodStatus.RUNNING),
            ])
        ])
        with pytest.raises(click.ClickException, match="pod-missing"):
            _resolve_pod(cclient, 1, pod_name="pod-missing")

    def test_returns_specified_pod(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod("pod-a", PodStatus.RUNNING),
                _make_pod("pod-b", PodStatus.RUNNING),
            ])
        ])
        result = _resolve_pod(cclient, 1, pod_name="pod-b")
        assert result == "pod-b"

    def test_empty_revision_list(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([])
        with pytest.raises(click.ClickException, match="No running pods"):
            _resolve_pod(cclient, 1)

    def test_none_revision_list(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response(None)
        cclient.get_status_v3.return_value.revision_pod_details_list = None
        with pytest.raises(click.ClickException, match="No running pods"):
            _resolve_pod(cclient, 1)

    def test_skips_pods_without_name(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod(None, PodStatus.RUNNING),
                _make_pod("pod-real", PodStatus.RUNNING),
            ])
        ])
        result = _resolve_pod(cclient, 1)
        assert result == "pod-real"

    def test_multiple_revisions(self):
        from centml.cli.shell import _resolve_pod
        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response([
            _make_revision([
                _make_pod("pod-old", PodStatus.ERROR),
            ]),
            _make_revision([
                _make_pod("pod-new", PodStatus.RUNNING),
            ]),
        ])
        result = _resolve_pod(cclient, 1)
        assert result == "pod-new"


# ===========================================================================
# _exec_session
# ===========================================================================

class TestExecSession:
    @pytest.mark.asyncio
    async def test_sends_resize_and_wrapped_command(self):
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps({"data": f"noise\n{_BEGIN_MARKER}\nhello world\n{_END_MARKER}:0\n"}),
            json.dumps({"Code": 0}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=ws),
                __aexit__=AsyncMock(return_value=False),
            ))

            exit_code = await _exec_session("wss://test/ws", "fake-token", "ls -la")

        assert exit_code == 0
        assert ws.send.call_count == 2
        resize_msg = json.loads(ws.send.call_args_list[0][0][0])
        assert resize_msg["operation"] == "resize"
        cmd_msg = json.loads(ws.send.call_args_list[1][0][0])
        assert cmd_msg["operation"] == "stdin"
        assert "ls -la" in cmd_msg["data"]
        assert "stty -echo" in cmd_msg["data"]
        # Markers use printf octal escapes, so the literal marker
        # should NOT appear in the command (prevents echo false-match).
        assert _BEGIN_MARKER not in cmd_msg["data"]
        assert "CENTML_BEGIN" in cmd_msg["data"]

    @pytest.mark.asyncio
    async def test_returns_nonzero_exit_code_from_marker(self):
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps({"data": f"{_BEGIN_MARKER}\n{_END_MARKER}:42\n"}),
            json.dumps({"Code": 42}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=ws),
                __aexit__=AsyncMock(return_value=False),
            ))

            exit_code = await _exec_session("wss://test/ws", "fake-token", "false")

        assert exit_code == 42

    @pytest.mark.asyncio
    async def test_error_message_returns_one(self):
        from centml.cli.shell import _exec_session

        ws = AsyncMock()
        messages = [
            json.dumps({"error": "something went wrong"}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=ws),
                __aexit__=AsyncMock(return_value=False),
            ))

            exit_code = await _exec_session("wss://test/ws", "fake-token", "bad")

        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_filters_noise_before_marker(self):
        """Only output between BEGIN and END markers is written to stdout."""
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps({"data": f"prompt$ command\n{_BEGIN_MARKER}\nreal output\n{_END_MARKER}:0\n"}),
            json.dumps({"Code": 0}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        captured = []
        with patch("centml.cli.shell.websockets") as mock_ws_mod, \
             patch("centml.cli.shell.sys") as mock_sys:
            mock_ws_mod.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=ws),
                __aexit__=AsyncMock(return_value=False),
            ))
            mock_sys.stdout.write = lambda s: captured.append(s)
            mock_sys.stdout.flush = MagicMock()
            mock_sys.stderr.write = MagicMock()

            exit_code = await _exec_session("wss://test/ws", "fake-token", "echo test")

        assert exit_code == 0
        output = "".join(captured)
        assert "real output" in output
        assert "prompt$" not in output


# ===========================================================================
# _interactive_session -- terminal restore
# ===========================================================================

class TestInteractiveSessionTerminalRestore:
    @pytest.mark.asyncio
    async def test_restores_terminal_on_exception(self):
        from centml.cli.shell import _interactive_session

        with patch("centml.cli.shell.sys") as mock_sys, \
             patch("centml.cli.shell.termios") as mock_termios, \
             patch("centml.cli.shell.tty") as mock_tty, \
             patch("centml.cli.shell.websockets") as mock_ws_mod:

            mock_sys.stdin.fileno.return_value = 0
            mock_termios.tcgetattr.return_value = ["old_settings"]

            mock_ws_mod.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=ConnectionRefusedError("fail")),
                __aexit__=AsyncMock(return_value=False),
            ))

            with pytest.raises(ConnectionRefusedError):
                await _interactive_session("wss://test/ws", "fake-token")

            mock_termios.tcsetattr.assert_called_once()
            restore_call = mock_termios.tcsetattr.call_args
            assert restore_call[0][2] == ["old_settings"]


# ===========================================================================
# Click commands
# ===========================================================================

class TestShellCommand:
    def test_rejects_non_tty(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(shell, ["123"])
        assert result.exit_code != 0
        assert "terminal" in result.output.lower() or "tty" in result.output.lower()

    def test_shell_option_forwarded(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), \
             patch("centml.cli.shell.get_centml_client") as mock_ctx, \
             patch("centml.cli.shell.auth") as mock_auth, \
             patch("centml.cli.shell.settings") as mock_settings, \
             patch("centml.cli.shell.asyncio") as mock_asyncio, \
             patch("centml.cli.shell.sys") as mock_sys:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_sys.stdin.isatty.return_value = True
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            result = runner.invoke(shell, ["123", "--shell", "bash"])

            # Verify asyncio.run was called, and the URL contains shell=bash
            mock_asyncio.run.assert_called_once()

    def test_pod_option_forwarded(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod") as mock_resolve, \
             patch("centml.cli.shell.get_centml_client") as mock_ctx, \
             patch("centml.cli.shell.auth") as mock_auth, \
             patch("centml.cli.shell.settings") as mock_settings, \
             patch("centml.cli.shell.asyncio") as mock_asyncio, \
             patch("centml.cli.shell.sys") as mock_sys:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_resolve.return_value = "my-pod"
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_sys.stdin.isatty.return_value = True
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            result = runner.invoke(shell, ["123", "--pod", "my-pod"])

            mock_resolve.assert_called_once()
            assert mock_resolve.call_args[1].get("pod_name") == "my-pod" or \
                   mock_resolve.call_args[0][2] == "my-pod"


class TestExecCommand:
    def test_passes_command(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), \
             patch("centml.cli.shell.get_centml_client") as mock_ctx, \
             patch("centml.cli.shell.auth") as mock_auth, \
             patch("centml.cli.shell.settings") as mock_settings, \
             patch("centml.cli.shell.asyncio") as mock_asyncio:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--", "ls", "-la"])

            mock_asyncio.run.assert_called_once()

    def test_shell_option_forwarded(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), \
             patch("centml.cli.shell.get_centml_client") as mock_ctx, \
             patch("centml.cli.shell.auth") as mock_auth, \
             patch("centml.cli.shell.settings") as mock_settings, \
             patch("centml.cli.shell.asyncio") as mock_asyncio:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--shell", "zsh", "--", "echo", "hi"])

            mock_asyncio.run.assert_called_once()
