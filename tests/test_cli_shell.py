"""Tests for centml.cli.shell -- thin Click command wrappers."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from centml.sdk.shell.exceptions import PodNotFoundError

# ===========================================================================
# _resolve_pod
# ===========================================================================


class TestResolvePod:
    def test_returns_valid_pod(self):
        from centml.cli.shell import _resolve_pod

        assert _resolve_pod(["pod-a", "pod-b"], "pod-b") == "pod-b"

    def test_raises_pod_not_found(self):
        from centml.cli.shell import _resolve_pod

        with pytest.raises(PodNotFoundError, match="pod-missing"):
            _resolve_pod(["pod-a"], "pod-missing")

    def test_error_lists_available_pods(self):
        from centml.cli.shell import _resolve_pod

        with pytest.raises(PodNotFoundError, match="pod-a, pod-b"):
            _resolve_pod(["pod-a", "pod-b"], "pod-c")


def _mock_client_ctx():
    """Return a patched get_centml_client context manager."""
    mock_ctx = MagicMock()
    mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
    return mock_ctx


@contextmanager
def _patch_deps(*, pods=None, tty=True):
    """Patch common CLI dependencies used by ``_connect_args``.

    If *pods* is provided, ``get_running_pods`` is also patched with that
    return value.  Yields a namespace exposing the mock objects that tests
    most often assert against.
    """
    with ExitStack() as stack:
        e = stack.enter_context
        e(patch("centml.cli.shell.get_centml_client", new_callable=_mock_client_ctx))
        if pods is not None:
            e(patch("centml.cli.shell.get_running_pods", return_value=pods))
        ns = SimpleNamespace(
            auth=e(patch("centml.cli.shell.auth")),
            settings=e(patch("centml.cli.shell.settings")),
            asyncio=e(patch("centml.cli.shell.asyncio")),
            sys=e(patch("centml.cli.shell.sys")),
            build_ws_url=e(patch("centml.cli.shell.build_ws_url")),
        )
        ns.auth.get_centml_token.return_value = "token"
        ns.settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
        ns.asyncio.run.return_value = 0
        ns.sys.stdin.isatty.return_value = tty
        ns.build_ws_url.return_value = "wss://test/ws"
        yield ns


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

        with _patch_deps(pods=["pod-a"]) as m:
            runner = CliRunner()
            runner.invoke(shell, ["123", "--shell", "bash"])
            m.asyncio.run.assert_called_once()

    def test_pod_option_forwarded(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=["my-pod"]) as m:
            runner = CliRunner()
            result = runner.invoke(shell, ["123", "--pod", "my-pod"])
            assert result.exit_code == 0
            m.build_ws_url.assert_called_once()
            assert "my-pod" in m.build_ws_url.call_args[0]

    def test_pod_not_found_error(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a"]):
            runner = CliRunner()
            result = runner.invoke(shell, ["123", "--pod", "bad-pod"])
            assert result.exit_code != 0
            assert "bad-pod" in result.output

    def test_interactive_selection_multiple_pods(self):
        """When multiple pods exist and no --pod, user picks from a list."""
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a", "pod-b", "pod-c"]) as m:
            runner = CliRunner()
            result = runner.invoke(shell, ["123"], input="2\n")
            assert result.exit_code == 0
            assert "pod-a" in result.output
            assert "pod-b" in result.output
            assert "pod-c" in result.output
            m.build_ws_url.assert_called_once()
            assert "pod-b" in m.build_ws_url.call_args[0]

    def test_first_pod_flag_skips_selection(self):
        """--first-pod auto-selects without prompting."""
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a", "pod-b"]) as m:
            runner = CliRunner()
            result = runner.invoke(shell, ["123", "--first-pod"])
            assert result.exit_code == 0
            assert "Select a pod" not in result.output
            m.build_ws_url.assert_called_once()
            assert "pod-a" in m.build_ws_url.call_args[0]

    def test_single_pod_auto_selects(self):
        """Single running pod is auto-selected without prompting."""
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-only"]) as m:
            runner = CliRunner()
            result = runner.invoke(shell, ["123"])
            assert result.exit_code == 0
            assert "Select a pod" not in result.output
            m.build_ws_url.assert_called_once()
            assert "pod-only" in m.build_ws_url.call_args[0]

    def test_no_running_pods_error(self):
        """Empty pod list raises an error."""
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with _patch_deps(pods=[]):
            runner = CliRunner()
            result = runner.invoke(shell, ["123"])
            assert result.exit_code != 0
            assert "No running pods" in result.output


class TestExecCommand:
    def test_passes_command(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a"]) as m:
            runner = CliRunner()
            runner.invoke(exec_cmd, ["123", "--", "ls", "-la"])
            m.asyncio.run.assert_called_once()

    def test_shell_option_forwarded(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a"]) as m:
            runner = CliRunner()
            runner.invoke(exec_cmd, ["123", "--shell", "zsh", "--", "echo", "hi"])
            m.asyncio.run.assert_called_once()

    def test_pod_not_found_error(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a"]):
            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--pod", "x", "--", "ls"])
            assert result.exit_code != 0
            assert "x" in result.output

    def test_interactive_selection_multiple_pods(self):
        """exec also prompts when multiple pods and no --pod."""
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a", "pod-b"]) as m:
            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--", "ls"], input="1\n")
            assert result.exit_code == 0
            assert "pod-a" in result.output
            assert "pod-b" in result.output
            m.build_ws_url.assert_called_once()
            assert "pod-a" in m.build_ws_url.call_args[0]

    def test_first_pod_flag_skips_selection(self):
        """--first-pod auto-selects for exec too."""
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with _patch_deps(pods=["pod-a", "pod-b"]) as m:
            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--first-pod", "--", "ls"])
            assert result.exit_code == 0
            assert "Select a pod" not in result.output
            m.build_ws_url.assert_called_once()
            assert "pod-a" in m.build_ws_url.call_args[0]
