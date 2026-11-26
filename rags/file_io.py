"""Bounded reads of operator-selected regular files without application setup."""
from __future__ import annotations

import os
import stat


class BoundedFileError(ValueError):
    """A content-free reason that callers can map to their own public errors."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def read_regular_bytes(path: str | os.PathLike[str], *, max_bytes: int) -> bytes:
    """Check and read the same descriptor, with a second bound after the read.

    O_NONBLOCK makes a concurrently substituted FIFO safe to reject with fstat;
    O_NOFOLLOW rejects a substituted leaf symlink on supported platforms.
    """
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("Read limit must be a nonnegative integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BoundedFileError("not_regular")
        if info.st_size > max_bytes:
            raise BoundedFileError("too_large")
        raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise BoundedFileError("read_limit_exceeded")
        return raw


def read_regular_text(path: str | os.PathLike[str], *, max_bytes: int) -> str:
    """Read bounded UTF-8 text, rejecting malformed encoding without replacement."""
    return read_regular_bytes(path, max_bytes=max_bytes).decode("utf-8")
