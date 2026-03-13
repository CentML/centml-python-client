"""Pyte terminal screen renderer -- converts pyte's in-memory buffer to ANSI."""

import pyte

_PYTE_FG_TO_SGR = {
    "default": "39",
    "black": "30",
    "red": "31",
    "green": "32",
    "brown": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "brightblack": "90",
    "brightred": "91",
    "brightgreen": "92",
    "brightbrown": "93",
    "brightblue": "94",
    "brightmagenta": "95",
    "brightcyan": "96",
    "brightwhite": "97",
}

_PYTE_BG_TO_SGR = {
    "default": "49",
    "black": "40",
    "red": "41",
    "green": "42",
    "brown": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    "brightblack": "100",
    "brightred": "101",
    "brightgreen": "102",
    "brightbrown": "103",
    "brightblue": "104",
    "brightmagenta": "105",
    "brightcyan": "106",
    "brightwhite": "107",
}


def color_sgr(color, is_bg=False):
    """Convert a pyte color value to an SGR parameter string."""
    table = _PYTE_BG_TO_SGR if is_bg else _PYTE_FG_TO_SGR
    if color in table:
        default_val = "49" if is_bg else "39"
        code = table[color]
        return code if code != default_val else ""
    # 6-char hex -> truecolor
    if len(color) == 6:
        try:
            r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
            prefix = "48" if is_bg else "38"
            return f"{prefix};2;{r};{g};{b}"
        except ValueError:
            return ""
    return ""


def char_to_sgr(char):
    """Build the ANSI SGR parameter string for a pyte Char's attributes."""
    parts = []
    if char.bold:
        parts.append("1")
    if char.italics:
        parts.append("3")
    if char.underscore:
        parts.append("4")
    if char.blink:
        parts.append("5")
    if char.reverse:
        parts.append("7")
    if char.strikethrough:
        parts.append("9")
    fg = color_sgr(char.fg, is_bg=False)
    if fg:
        parts.append(fg)
    bg = color_sgr(char.bg, is_bg=True)
    if bg:
        parts.append(bg)
    return ";".join(parts)


def render_dirty(screen, output):
    """Render only the dirty lines from the pyte Screen to the terminal.

    Args:
        screen: pyte.Screen instance.
        output: Writable binary stream (e.g. sys.stdout.buffer).
    """
    parts = []
    for row in sorted(screen.dirty):
        # Position cursor at row (1-based), column 1; clear line.
        parts.append(f"\033[{row + 1};1H\033[2K")
        prev_sgr = ""
        line_chars = []
        for col in range(screen.columns):
            char = screen.buffer[row][col]
            if char.data == "":
                continue
            sgr = char_to_sgr(char)
            if sgr != prev_sgr:
                line_chars.append(f"\033[0m\033[{sgr}m" if sgr else "\033[0m")
                prev_sgr = sgr
            line_chars.append(char.data)
        text = "".join(line_chars).rstrip()
        parts.append(text)
    # Reset attributes, position cursor.
    parts.append("\033[0m")
    parts.append(f"\033[{screen.cursor.y + 1};{screen.cursor.x + 1}H")
    if screen.cursor.hidden:
        parts.append("\033[?25l")
    else:
        parts.append("\033[?25h")
    screen.dirty.clear()
    output.write("".join(parts).encode("utf-8"))
    output.flush()


def pyte_extract_text(line_stream, line_screen, text):
    """Feed text through a single-row pyte screen and return visible characters.

    More robust than regex ANSI stripping: pyte interprets all VT100/VT220
    sequences including OSC, cursor repositioning, and truecolor escapes.
    """
    line_screen.reset()
    line_stream.feed(text)
    return "".join(line_screen.buffer[0][col].data for col in range(line_screen.columns)).rstrip()
