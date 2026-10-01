"""Lossless, deterministic encoding for complete physical SVG exports."""

from __future__ import annotations

import gzip
from pathlib import Path


def encode_svgz(source: str) -> bytes:
    """Preserve the complete UTF-8 document in the registered SVGZ format."""
    return gzip.compress(source.encode("utf-8"), compresslevel=6, mtime=0)


def read_svgz(path: str | Path) -> str:
    """Read a canonical SVGZ document, validating its gzip stream and UTF-8."""
    return gzip.decompress(Path(path).read_bytes()).decode("utf-8")
