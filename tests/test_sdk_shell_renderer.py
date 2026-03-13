"""Tests for centml.sdk.shell.renderer -- pyte rendering utilities."""

import io

import pyte

from centml.sdk.shell.renderer import char_to_sgr, color_sgr, pyte_extract_text, render_dirty

# ===========================================================================
# color_sgr
# ===========================================================================


class TestColorSgr:
    def test_named_fg_color(self):
        assert color_sgr("red", is_bg=False) == "31"

    def test_named_bg_color(self):
        assert color_sgr("blue", is_bg=True) == "44"

    def test_default_fg_returns_empty(self):
        assert color_sgr("default", is_bg=False) == ""

    def test_default_bg_returns_empty(self):
        assert color_sgr("default", is_bg=True) == ""

    def test_hex_truecolor_fg(self):
        assert color_sgr("ff0000", is_bg=False) == "38;2;255;0;0"

    def test_hex_truecolor_bg(self):
        assert color_sgr("00ff00", is_bg=True) == "48;2;0;255;0"

    def test_invalid_hex_returns_empty(self):
        assert color_sgr("zzzzzz", is_bg=False) == ""

    def test_unknown_name_returns_empty(self):
        assert color_sgr("nope", is_bg=False) == ""


# ===========================================================================
# char_to_sgr
# ===========================================================================


class TestCharToSgr:
    def test_default_attrs_returns_empty(self):
        char = pyte.screens.Char(" ", "default", "default", False, False, False, False, False, False)
        assert char_to_sgr(char) == ""

    def test_bold_red_fg(self):
        char = pyte.screens.Char("x", "red", "default", True, False, False, False, False, False)
        sgr = char_to_sgr(char)
        assert "1" in sgr.split(";")
        assert "31" in sgr.split(";")

    def test_bg_color(self):
        char = pyte.screens.Char("x", "default", "blue", False, False, False, False, False, False)
        sgr = char_to_sgr(char)
        assert "44" in sgr.split(";")

    def test_256_color_fg(self):
        char = pyte.screens.Char("x", "ff0000", "default", False, False, False, False, False, False)
        sgr = char_to_sgr(char)
        assert "38;2;255;0;0" in sgr

    def test_combined_attrs(self):
        char = pyte.screens.Char("x", "green", "white", True, True, True, False, False, False)
        sgr = char_to_sgr(char)
        parts = sgr.split(";")
        assert "1" in parts  # bold
        assert "3" in parts  # italics
        assert "4" in parts  # underscore
        assert "32" in parts  # green fg
        assert "47" in parts  # white bg


# ===========================================================================
# render_dirty
# ===========================================================================


class TestRenderDirty:
    def test_renders_simple_text(self):
        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        screen.dirty.clear()
        stream.feed("hello")
        buf = io.BytesIO()
        render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        assert "hello" in output
        assert len(screen.dirty) == 0

    def test_clears_dirty_after_render(self):
        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        screen.dirty.clear()
        stream.feed("test")
        assert len(screen.dirty) > 0
        render_dirty(screen, io.BytesIO())
        assert len(screen.dirty) == 0

    def test_cursor_position_in_output(self):
        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        stream.feed("abc")
        buf = io.BytesIO()
        render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        # Cursor should be at row 1, col 4 (1-based: after "abc")
        assert "\033[1;4H" in output

    def test_renders_only_dirty_lines(self):
        screen = pyte.Screen(40, 5)
        stream = pyte.Stream(screen)
        stream.feed("line0\r\nline1\r\nline2")
        # Render to clear dirty
        render_dirty(screen, io.BytesIO())
        # Now modify only line 0
        stream.feed("\033[1;1Hchanged")
        buf = io.BytesIO()
        render_dirty(screen, buf)
        output = buf.getvalue().decode("utf-8")
        assert "changed" in output
        # line1 and line2 should NOT be re-rendered
        assert "line1" not in output
        assert "line2" not in output


# ===========================================================================
# pyte_extract_text
# ===========================================================================


class TestPyteExtractText:
    def test_strips_ansi(self):
        screen = pyte.Screen(80, 1)
        stream = pyte.Stream(screen)
        result = pyte_extract_text(stream, screen, "\x1b[32mhello\x1b[0m")
        assert result == "hello"

    def test_plain_text(self):
        screen = pyte.Screen(80, 1)
        stream = pyte.Stream(screen)
        result = pyte_extract_text(stream, screen, "plain text")
        assert result == "plain text"
