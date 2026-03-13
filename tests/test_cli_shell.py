"""Tests for centml.cli.shell -- thin Click command wrappers."""

from unittest.mock import MagicMock, patch

import click
import pytest

from centml.sdk.shell.exceptions import NoPodAvailableError, PodNotFoundError


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

        with patch("centml.cli.shell.resolve_pod", return_value=("pod-a", None)), patch(
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

        with patch("centml.cli.shell.resolve_pod") as mock_resolve, patch(
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
            mock_resolve.return_value = ("my-pod", None)
            mock_auth.get_centml_token.return_value = "token"
            mock_settings.CENTML_PLATFORM_API_URL = "https://api.centml.com"
            mock_sys.stdin.isatty.return_value = True
            mock_asyncio.run.return_value = 0

            runner = CliRunner()
            runner.invoke(shell, ["123", "--pod", "my-pod"])

            mock_resolve.assert_called_once()

    def test_shell_error_converts_to_click_exception(self):
        from centml.cli.shell import shell
        from click.testing import CliRunner

        with patch("centml.cli.shell.resolve_pod", side_effect=NoPodAvailableError("No running pods found")), patch(
            "centml.cli.shell.get_centml_client"
        ) as mock_ctx, patch("centml.cli.shell.setup_debug_log"), patch("centml.cli.shell.sys") as mock_sys:

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_sys.stdin.isatty.return_value = True

            runner = CliRunner()
            result = runner.invoke(shell, ["123"])

            assert result.exit_code != 0
            assert "No running pods" in result.output


class TestExecCommand:
    def test_passes_command(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch("centml.cli.shell.resolve_pod", return_value=("pod-a", None)), patch(
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

        with patch("centml.cli.shell.resolve_pod", return_value=("pod-a", None)), patch(
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

    def test_shell_error_converts_to_click_exception(self):
        from centml.cli.shell import exec_cmd
        from click.testing import CliRunner

        with patch(
            "centml.cli.shell.resolve_pod", side_effect=PodNotFoundError("Pod 'x' not found")
        ), patch("centml.cli.shell.get_centml_client") as mock_ctx, patch("centml.cli.shell.setup_debug_log"):

            mock_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            runner = CliRunner()
            result = runner.invoke(exec_cmd, ["123", "--", "ls"])

            assert result.exit_code != 0
            assert "Pod 'x' not found" in result.output
