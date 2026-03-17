"""Tests for _pyte_extract_text in centml.sdk.shell.session."""

import pyte

from centml.sdk.shell.session import _pyte_extract_text


class TestPyteExtractText:
    def test_strips_ansi(self):
        screen = pyte.Screen(80, 1)
        stream = pyte.Stream(screen)
        result = _pyte_extract_text(stream, screen, "\x1b[32mhello\x1b[0m")
        assert result == "hello"

    def test_plain_text(self):
        screen = pyte.Screen(80, 1)
        stream = pyte.Stream(screen)
        result = _pyte_extract_text(stream, screen, "plain text")
        assert result == "plain text"
