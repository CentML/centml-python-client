"""Tests for centml.sdk.utils.config_file.load_config_file_mount."""

import pytest

from centml.sdk.utils.config_file import load_config_file_mount


def test_default_filename_from_basename(tmp_path):
    src = tmp_path / "nginx.conf"
    src.write_text("server { listen 80; }\n")

    mount = load_config_file_mount(str(src), "/etc/nginx/conf.d")

    assert mount.filename == "nginx.conf"
    assert mount.mount_path == "/etc/nginx/conf.d"
    assert mount.content == "server { listen 80; }\n"


def test_explicit_filename_overrides_basename(tmp_path):
    src = tmp_path / "local.txt"
    src.write_text("payload")

    mount = load_config_file_mount(str(src), "/app/etc", filename="remote.conf")

    assert mount.filename == "remote.conf"
    assert mount.mount_path == "/app/etc"
    assert mount.content == "payload"


def test_utf8_multibyte_content_roundtrips(tmp_path):
    src = tmp_path / "i18n.conf"
    src.write_text("配置内容 = 测试\n", encoding="utf-8")

    mount = load_config_file_mount(str(src), "/etc/app")

    assert mount.content == "配置内容 = 测试\n"


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config_file_mount(str(tmp_path / "does-not-exist.conf"), "/etc/x")


def test_preserves_crlf_line_endings(tmp_path):
    # Windows-authored configs use \r\n; the helper must not silently
    # normalize them to \n when uploading to the server.
    src = tmp_path / "windows.conf"
    src.write_bytes(b"line1\r\nline2\r\n")

    mount = load_config_file_mount(str(src), "/etc/app")

    assert mount.content == "line1\r\nline2\r\n"
