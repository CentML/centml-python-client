"""Tests for centml.cli.shell -- CLI terminal access commands."""

import asyncio
import io
import json
import os
import signal
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pyte
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


class TestBuildWsUrl:
    def test_https_to_wss(self):
        from centml.cli.shell import _build_ws_url

        url = _build_ws_url("https://api.centml.com", 123, "my-pod-abc")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "wss"
        assert parsed.netloc == "api.centml.com"

    def test_http_to_ws(self):
        from centml.cli.shell import _build_ws_url

        url = _build_ws_url("http://localhost:16000", 42, "pod-1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "ws"
        assert parsed.netloc == "localhost:16000"

    def test_contains_deployment_id_and_pod(self):
        from centml.cli.shell import _build_ws_url

        url = _build_ws_url("https://api.centml.com", 99, "pod-xyz")
        assert "/deployments/99/terminal" in url
        assert "pod=pod-xyz" in url

    def test_with_shell(self):
        from centml.cli.shell import _build_ws_url

        url = _build_ws_url("https://api.centml.com", 1, "p", shell_type="bash")
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
        cclient.get_status_v3.return_value = _make_status_response(
            [
                _make_revision(
                    [
                        _make_pod("pod-a", PodStatus.RUNNING),
                        _make_pod("pod-b", PodStatus.RUNNING),
                    ]
                )
            ]
        )
        result = _resolve_pod(cclient, 1)
        assert result == "pod-a"

    def test_raises_no_running_pods(self):
        from centml.cli.shell import _resolve_pod

        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response(
            [_make_revision([_make_pod("pod-err", PodStatus.ERROR)])]
        )
        with pytest.raises(click.ClickException, match="No running pods"):
            _resolve_pod(cclient, 1)

    def test_raises_specified_pod_not_found(self):
        from centml.cli.shell import _resolve_pod

        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response(
            [_make_revision([_make_pod("pod-a", PodStatus.RUNNING)])]
        )
        with pytest.raises(click.ClickException, match="pod-missing"):
            _resolve_pod(cclient, 1, pod_name="pod-missing")

    def test_returns_specified_pod(self):
        from centml.cli.shell import _resolve_pod

        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response(
            [
                _make_revision(
                    [
                        _make_pod("pod-a", PodStatus.RUNNING),
                        _make_pod("pod-b", PodStatus.RUNNING),
                    ]
                )
            ]
        )
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
        cclient.get_status_v3.return_value = _make_status_response(
            [
                _make_revision(
                    [
                        _make_pod(None, PodStatus.RUNNING),
                        _make_pod("pod-real", PodStatus.RUNNING),
                    ]
                )
            ]
        )
        result = _resolve_pod(cclient, 1)
        assert result == "pod-real"

    def test_multiple_revisions(self):
        from centml.cli.shell import _resolve_pod

        cclient = MagicMock()
        cclient.get_status_v3.return_value = _make_status_response(
            [
                _make_revision([_make_pod("pod-old", PodStatus.ERROR)]),
                _make_revision([_make_pod("pod-new", PodStatus.RUNNING)]),
            ]
        )
        result = _resolve_pod(cclient, 1)
        assert result == "pod-new"


# ===========================================================================
# _exec_session
# ===========================================================================


class TestExecSession:
    def test_sends_resize_and_wrapped_command(self):
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps(
                {"data": f"noise\n{_BEGIN_MARKER}\nhello world\n{_END_MARKER}:0\n"}
            ),
            json.dumps({"Code": 0}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            exit_code = asyncio.run(
                _exec_session("wss://test/ws", "fake-token", "ls -la")
            )

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

    def test_returns_nonzero_exit_code_from_marker(self):
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps({"data": f"{_BEGIN_MARKER}\n{_END_MARKER}:42\n"}),
            json.dumps({"Code": 42}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            exit_code = asyncio.run(
                _exec_session("wss://test/ws", "fake-token", "false")
            )

        assert exit_code == 42

    def test_error_message_returns_one(self):
        from centml.cli.shell import _exec_session

        ws = AsyncMock()
        messages = [json.dumps({"error": "something went wrong"})]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            exit_code = asyncio.run(_exec_session("wss://test/ws", "fake-token", "bad"))

        assert exit_code == 1

    def test_filters_noise_before_marker(self):
        """Only output between BEGIN and END markers is written to stdout."""
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps(
                {
                    "data": f"prompt$ command\n{_BEGIN_MARKER}\nreal output\n{_END_MARKER}:0\n"
                }
            ),
            json.dumps({"Code": 0}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        captured = []
        with patch("centml.cli.shell.websockets") as mock_ws_mod, patch(
            "centml.cli.shell.sys"
        ) as mock_sys:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )
            mock_sys.stdout.write = lambda s: captured.append(s)
            mock_sys.stdout.flush = MagicMock()
            mock_sys.stderr.write = MagicMock()

            exit_code = asyncio.run(
                _exec_session("wss://test/ws", "fake-token", "echo test")
            )

        assert exit_code == 0
        output = "".join(captured)
        assert "real output" in output
        assert "prompt$" not in output

    def test_connection_closed_returns_zero(self):
        """Graceful exit when server closes connection without Code message."""
        from centml.cli.shell import _exec_session

        import websockets as _ws_lib

        ws = AsyncMock()

        async def _raise_closed():
            yield json.dumps({"data": "partial\n"})
            raise _ws_lib.ConnectionClosed(None, None)

        ws.__aiter__ = MagicMock(return_value=_raise_closed())

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )
            mock_ws_mod.ConnectionClosed = _ws_lib.ConnectionClosed

            exit_code = asyncio.run(
                _exec_session("wss://test/ws", "fake-token", "exit")
            )

        assert exit_code == 0

    def test_sets_zero_close_timeout_after_done(self):
        """After END marker, close_timeout should be 0 to avoid waiting for server close."""
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        messages = [
            json.dumps({"data": f"{_BEGIN_MARKER}\nhello\n{_END_MARKER}:0\n"}),
        ]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))
        ws.close_timeout = 2

        with patch("centml.cli.shell.websockets") as mock_ws_mod:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            asyncio.run(_exec_session("wss://test/ws", "fake-token", "echo hello"))

        assert ws.close_timeout == 0

    def test_handles_ansi_around_markers(self):
        """Markers wrapped in ANSI codes are still detected via pyte."""
        from centml.cli.shell import _exec_session, _BEGIN_MARKER, _END_MARKER

        ws = AsyncMock()
        # Markers surrounded by ANSI color codes.
        data = f"\x1b[32m{_BEGIN_MARKER}\x1b[0m\noutput\n\x1b[32m{_END_MARKER}:0\x1b[0m\n"
        messages = [json.dumps({"data": data}), json.dumps({"Code": 0})]
        ws.__aiter__ = MagicMock(return_value=_async_iter_from_list(messages))

        captured = []
        with patch("centml.cli.shell.websockets") as mock_ws_mod, patch(
            "centml.cli.shell.sys"
        ) as mock_sys:
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )
            mock_sys.stdout.write = lambda s: captured.append(s)
            mock_sys.stdout.flush = MagicMock()
            mock_sys.stderr.write = MagicMock()

            exit_code = asyncio.run(
                _exec_session("wss://test/ws", "fake-token", "echo test")
            )

        assert exit_code == 0
        output = "".join(captured)
        assert "output" in output


# ===========================================================================
# _interactive_session -- terminal restore
# ===========================================================================


class TestInteractiveSessionTerminalRestore:
    def test_restores_terminal_on_exception(self):
        from centml.cli.shell import _interactive_session

        with patch("centml.cli.shell.sys") as mock_sys, patch(
            "centml.cli.shell.termios"
        ) as mock_termios, patch("centml.cli.shell.tty"), patch(
            "centml.cli.shell.websockets"
        ) as mock_ws_mod:

            mock_sys.stdin.fileno.return_value = 0
            mock_sys.stdout.buffer = io.BytesIO()
            mock_termios.tcgetattr.return_value = ["old_settings"]

            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=ConnectionRefusedError("fail")),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            with pytest.raises(ConnectionRefusedError):
                asyncio.run(_interactive_session("wss://test/ws", "fake-token"))

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

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), patch(
            "centml.cli.shell.get_centml_client"
        ) as mock_ctx, patch("centml.cli.shell.auth") as mock_auth, patch(
            "centml.cli.shell.settings"
        ) as mock_settings, patch(
            "centml.cli.shell.asyncio"
        ) as mock_asyncio, patch(
            "centml.cli.shell.sys"
        ) as mock_sys:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_sys.stdin.isatty.return_value = True
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            runner.invoke(shell, ["123", "--shell", "bash"])

            mock_asyncio.run.assert_called_once()

    def test_pod_option_forwarded(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod") as mock_resolve, patch(
            "centml.cli.shell.get_centml_client"
        ) as mock_ctx, patch("centml.cli.shell.auth") as mock_auth, patch(
            "centml.cli.shell.settings"
        ) as mock_settings, patch(
            "centml.cli.shell.asyncio"
        ) as mock_asyncio, patch(
            "centml.cli.shell.sys"
        ) as mock_sys:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_resolve.return_value = "my-pod"
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_sys.stdin.isatty.return_value = True
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            runner.invoke(shell, ["123", "--pod", "my-pod"])

            mock_resolve.assert_called_once()
            assert (
                mock_resolve.call_args[1].get("pod_name") == "my-pod"
                or mock_resolve.call_args[0][2] == "my-pod"
            )


class TestExecCommand:
    def test_passes_command(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), patch(
            "centml.cli.shell.get_centml_client"
        ) as mock_ctx, patch("centml.cli.shell.auth") as mock_auth, patch(
            "centml.cli.shell.settings"
        ) as mock_settings, patch(
            "centml.cli.shell.asyncio"
        ) as mock_asyncio:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            runner.invoke(exec_cmd, ["123", "--", "ls", "-la"])

            mock_asyncio.run.assert_called_once()

    def test_shell_option_forwarded(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch("centml.cli.shell._resolve_pod", return_value="pod-a"), patch(
            "centml.cli.shell.get_centml_client"
        ) as mock_ctx, patch("centml.cli.shell.auth") as mock_auth, patch(
            "centml.cli.shell.settings"
        ) as mock_settings, patch(
            "centml.cli.shell.asyncio"
        ) as mock_asyncio:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            runner.invoke(exec_cmd, ["123", "--shell", "zsh", "--", "echo", "hi"])

            mock_asyncio.run.assert_called_once()


# ===========================================================================
# pyte renderer: _char_to_sgr
# ===========================================================================


class TestCharToSgr:
    def test_default_attrs_returns_empty(self):
        from centml.cli.shell import _char_to_sgr

        char = pyte.screens.Char(
            " ", "default", "default", False, False, False, False, False, False
        )
        assert _char_to_sgr(char) == ""

    def test_bold_red_fg(self):
        from centml.cli.shell import _char_to_sgr

        char = pyte.screens.Char(
            "x", "red", "default", True, False, False, False, False, False
        )
        sgr = _char_to_sgr(char)
        assert "1" in sgr.split(";")
        assert "31" in sgr.split(";")

    def test_bg_color(self):
        from centml.cli.shell import _char_to_sgr

        char = pyte.screens.Char(
            "x", "default", "blue", False, False, False, False, False, False
        )
        sgr = _char_to_sgr(char)
        assert "44" in sgr.split(";")

    def test_256_color_fg(self):
        from centml.cli.shell import _char_to_sgr

        char = pyte.screens.Char(
            "x", "ff0000", "default", False, False, False, False, False, False
        )
        sgr = _char_to_sgr(char)
        assert "38;2;255;0;0" in sgr

    def test_combined_attrs(self):
        from centml.cli.shell import _char_to_sgr

        char = pyte.screens.Char(
            "x", "green", "white", True, True, True, False, False, False
        )
        sgr = _char_to_sgr(char)
        parts = sgr.split(";")
        assert "1" in parts  # bold
        assert "3" in parts  # italics
        assert "4" in parts  # underscore
        assert "32" in parts  # green fg
        assert "47" in parts  # white bg


# ===========================================================================
# pyte renderer: _render_dirty
# ===========================================================================


class TestRenderDirty:
    def test_renders_simple_text(self):
        from centml.cli.shell import _render_dirty

        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        screen.dirty.clear()
        stream.feed("hello")
        buf = io.BytesIO()
        _render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        assert "hello" in output
        assert len(screen.dirty) == 0

    def test_clears_dirty_after_render(self):
        from centml.cli.shell import _render_dirty

        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        screen.dirty.clear()
        stream.feed("test")
        assert len(screen.dirty) > 0
        _render_dirty(screen, io.BytesIO())
        assert len(screen.dirty) == 0

    def test_cursor_position_in_output(self):
        from centml.cli.shell import _render_dirty

        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        stream.feed("abc")
        buf = io.BytesIO()
        _render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        # Cursor should be at row 1, col 4 (1-based: after "abc")
        assert "\033[1;4H" in output

    def test_renders_only_dirty_lines(self):
        from centml.cli.shell import _render_dirty

        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        stream.feed("line0\r\nline1\r\nline2")
        # Render to clear dirty
        _render_dirty(screen, io.BytesIO())
        # Now modify only line 0
        stream.feed("\033[1;1Hchanged")
        buf = io.BytesIO()
        _render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        assert "changed" in output
        # line1 and line2 should NOT be re-rendered
        assert "line1" not in output
        assert "line2" not in output


# ===========================================================================
# _forward_io -- exit detection and shutdown
# ===========================================================================


class TestForwardIo:
    """Tests for _forward_io exit detection via idle timeout.

    Uses a real pipe fd so ``loop.add_reader`` works without OS errors.
    """

    def _run_forward_io(self, ws, shutdown=None):
        """Helper: run _forward_io with a real pipe fd standing in for stdin."""
        from centml.cli.shell import _forward_io

        import websockets as _ws_lib

        screen = pyte.Screen(80, 24)
        stream = pyte.Stream(screen)
        if shutdown is None:
            shutdown = asyncio.Event()

        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            with patch("centml.cli.shell.sys") as mock_sys, patch(
                "centml.cli.shell.websockets"
            ) as mock_ws_mod:
                mock_sys.stdin.fileno.return_value = read_fd
                mock_sys.stdin.buffer.read1 = lambda n: b""
                mock_sys.stdout.buffer = io.BytesIO()
                mock_ws_mod.ConnectionClosed = _ws_lib.ConnectionClosed

                return asyncio.run(_forward_io(ws, screen, stream, shutdown))
        finally:
            os.close(read_fd)

    def test_connection_closed_returns_zero(self):
        """ConnectionClosed returns 0."""
        import websockets as _ws_lib

        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=_ws_lib.ConnectionClosed(None, None))

        assert self._run_forward_io(ws) == 0

    def test_exit_echo_at_end_exits_immediately(self):
        """'exit\\r\\n' at end of data (no trailing prompt) exits immediately."""
        ws = AsyncMock()
        ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"data": "\r\n\x1b[?2004l\rexit\r\n"}),
                # Should never be called -- _read_ws returns before this.
                json.dumps({"data": "should not reach"}),
            ]
        )

        assert self._run_forward_io(ws) == 0
        # Only one recv call -- exited immediately after exit echo.
        assert ws.recv.call_count == 1

    def test_exit_echo_with_prompt_continues(self):
        """'exit\\r\\n' followed by a new prompt is not a real exit."""
        import websockets as _ws_lib

        ws = AsyncMock()
        ws.recv = AsyncMock(
            side_effect=[
                # ``echo exit`` -- exit echo with prompt trailing.
                json.dumps({"data": "\r\n\x1b[?2004l\rexit\r\n\x1b[?2004huser@host:~$ "}),
                _ws_lib.ConnectionClosed(None, None),
            ]
        )

        assert self._run_forward_io(ws) == 0
        # Both recv calls made -- did not exit after the first message.
        assert ws.recv.call_count == 2

    def test_normal_data_no_early_exit(self):
        """Data without 'exit\\r\\n' does not trigger early exit."""
        import websockets as _ws_lib

        ws = AsyncMock()
        ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"data": "hello\r\n"}),
                _ws_lib.ConnectionClosed(None, None),
            ]
        )

        assert self._run_forward_io(ws) == 0

    def test_shutdown_event_exits(self):
        """shutdown event causes _forward_io to exit."""
        from centml.cli.shell import _forward_io

        import websockets as _ws_lib

        ws = AsyncMock()

        # recv that blocks until cancelled (simulates open WS with no data)
        async def _block_recv():
            await asyncio.sleep(999)

        ws.recv = _block_recv

        screen = pyte.Screen(80, 24)
        stream = pyte.Stream(screen)
        shutdown = asyncio.Event()

        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            with patch("centml.cli.shell.sys") as mock_sys, patch(
                "centml.cli.shell.websockets"
            ) as mock_ws_mod:
                mock_sys.stdin.fileno.return_value = read_fd
                mock_sys.stdin.buffer.read1 = lambda n: b""
                mock_sys.stdout.buffer = io.BytesIO()
                mock_ws_mod.ConnectionClosed = _ws_lib.ConnectionClosed

                async def _run():
                    async def _set_shutdown():
                        await asyncio.sleep(0.1)
                        shutdown.set()

                    asyncio.create_task(_set_shutdown())
                    return await _forward_io(ws, screen, stream, shutdown)

                assert asyncio.run(_run()) == 0
        finally:
            os.close(read_fd)


# ===========================================================================
# _interactive_session -- signal handling
# ===========================================================================


class TestInteractiveSessionSignals:
    """Tests for SIGTERM/SIGHUP restoring terminal settings."""

    def test_sigterm_restores_terminal(self):
        from centml.cli.shell import _interactive_session

        signal_handlers = {}

        def _fake_add_signal_handler(sig, handler):
            signal_handlers[sig] = handler

        def _fake_remove_signal_handler(sig):
            signal_handlers.pop(sig, None)

        async def _fake_forward_io(ws, screen, stream, shutdown):
            if signal.SIGTERM in signal_handlers:
                signal_handlers[signal.SIGTERM]()
            return 0

        with patch("centml.cli.shell.sys") as mock_sys, patch(
            "centml.cli.shell.termios"
        ) as mock_termios, patch("centml.cli.shell.tty"), patch(
            "centml.cli.shell.websockets"
        ) as mock_ws_mod, patch(
            "centml.cli.shell._forward_io", side_effect=_fake_forward_io
        ):
            mock_sys.stdin.fileno.return_value = 0
            mock_sys.stdout.buffer = io.BytesIO()
            mock_termios.tcgetattr.return_value = ["old_settings"]

            mock_ws = AsyncMock()
            mock_ws_mod.connect = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_ws),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            def _patched_run(coro):
                loop = asyncio.new_event_loop()
                loop.add_signal_handler = _fake_add_signal_handler
                loop.remove_signal_handler = _fake_remove_signal_handler
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            with patch("centml.cli.shell.asyncio") as mock_asyncio_mod:
                mock_asyncio_mod.get_running_loop.return_value = MagicMock(
                    add_signal_handler=_fake_add_signal_handler,
                    remove_signal_handler=_fake_remove_signal_handler,
                )
                mock_asyncio_mod.Event = asyncio.Event
                mock_asyncio_mod.create_task = asyncio.ensure_future

                _patched_run(
                    _interactive_session("wss://test/ws", "fake-token")
                )

            mock_termios.tcsetattr.assert_called_once()
            assert mock_termios.tcsetattr.call_args[0][2] == ["old_settings"]
