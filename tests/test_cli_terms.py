from unittest.mock import patch

from click.testing import CliRunner

from centml.cli.main import cli
from centml.sdk.config import settings

TOS_URL = settings.CENTML_TERMS_OF_SERVICE_URL


def test_version_shows_terms_url():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert TOS_URL in result.output


def test_login_shows_terms_url_when_not_authenticated():
    with patch("centml.cli.login.auth.load_centml_cred", return_value=None):
        result = CliRunner().invoke(cli, ["login"], input="n\n")
    assert TOS_URL in result.output
    assert "Terms of Service" in result.output


def test_login_with_stored_credentials_skips_terms_notice():
    with patch("centml.cli.login.auth.load_centml_cred", return_value={"refresh_token": "x"}), patch(
        "centml.cli.login.auth.refresh_centml_token", return_value=True
    ):
        result = CliRunner().invoke(cli, ["login"])
    assert "Login successful" in result.output
    assert TOS_URL not in result.output
