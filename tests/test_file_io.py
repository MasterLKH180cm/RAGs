"""Regression coverage for bounded descriptor ownership and file replacement."""
import errno
import os

import pytest

from rags.file_io import BoundedFileError, read_regular_bytes, read_regular_text


pytestmark = pytest.mark.unit


def test_text_budget_counts_utf8_bytes_and_rejects_invalid_encoding(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("資料", encoding="utf-8")
    assert read_regular_text(source, max_bytes=6) == "資料"
    with pytest.raises(BoundedFileError, match="too_large"):
        read_regular_text(source, max_bytes=5)
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        read_regular_text(source, max_bytes=6)


def test_file_growth_after_descriptor_stat_is_still_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"initial")
    original = os.fstat

    def grow(fd):
        info = original(fd)
        with source.open("ab") as stream:
            stream.write(b"x" * 100)
        return info

    monkeypatch.setattr(os, "fstat", grow)
    with pytest.raises(BoundedFileError, match="read_limit_exceeded"):
        read_regular_bytes(source, max_bytes=10)


def test_replaced_path_does_not_change_the_opened_input(tmp_path, monkeypatch):
    source, replacement = tmp_path / "source.txt", tmp_path / "replacement.txt"
    source.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    original = os.open

    def replace_after_open(path, flags):
        fd = original(path, flags)
        source.unlink()
        source.symlink_to(replacement)
        return fd

    monkeypatch.setattr(os, "open", replace_after_open)
    assert read_regular_bytes(source, max_bytes=100) == b"original"


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_swapped_special_files_are_rejected_without_blocking(tmp_path, monkeypatch, replacement):
    source, other = tmp_path / "source.txt", tmp_path / "other.txt"
    source.write_bytes(b"original")
    other.write_bytes(b"private-content")
    original = os.open

    def replace_before_open(path, flags):
        source.unlink()
        if replacement == "symlink":
            source.symlink_to(other)
        else:
            os.mkfifo(source)
            # Fail before opening if the nonblocking guard ever regresses.
            assert flags & os.O_NONBLOCK
        return original(path, flags)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises((BoundedFileError, OSError)):
        read_regular_bytes(source, max_bytes=100)


def test_rejected_file_descriptor_is_closed(tmp_path, monkeypatch):
    source = tmp_path / "too-large.txt"
    source.write_bytes(b"too large")
    opened = []
    original = os.open

    def record(path, flags):
        fd = original(path, flags)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", record)
    with pytest.raises(BoundedFileError, match="too_large"):
        read_regular_bytes(source, max_bytes=1)
    with pytest.raises(OSError) as error:
        os.fstat(opened[0])
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_read_budget_is_rejected_before_opening(tmp_path, limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        read_regular_bytes(tmp_path / "missing", max_bytes=limit)
