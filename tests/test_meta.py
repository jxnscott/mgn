"""Tests for ``mgn.meta``."""

import json
from pathlib import Path
from typing import Any

import pytest

from mgn.meta import _normalize_dtype, sanitize_meta_json_dtypes


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("<dtype: 'float32'>", "float32"),
        ("<dtype: 'int32'>", "int32"),
        ("float32", "float32"),
    ],
)
def test_normalize_dtype(dtype: str, expected: str) -> None:
    """Unwraps a stringified ``tf.DType`` repr, and passes plain names through.

    Args:
        dtype: The raw dtype string, as it would appear in meta.json.
        expected: The plain dtype name ``_normalize_dtype`` should return.
    """
    assert _normalize_dtype(dtype=dtype) == expected


def _write_meta(
    *, path: Path, dtypes: dict[str, str], trajectory_length: int, shape: list[int]
) -> dict[str, Any]:
    """Writes a minimal meta.json fixture with one feature per dtype given.

    Args:
        path: File path to write the meta.json to.
        dtypes: Maps feature name to the (possibly malformed) dtype string
            to give that feature.
        trajectory_length: Value to write as the top-level ``trajectory_length``.
        shape: Shape to give each feature.

    Returns:
        The metadata dict that was written, for comparison in assertions.
    """
    meta = {
        "trajectory_length": trajectory_length,
        "features": {
            name: {"type": "dynamic", "shape": shape, "dtype": dtype}
            for name, dtype in dtypes.items()
        },
    }
    path.write_text(json.dumps(meta))
    return meta


def test_sanitize_meta_json_dtypes_normalizes_in_place(tmp_path: Path) -> None:
    """Rewrites every feature's dtype to its plain name, leaving the rest untouched."""
    meta_path = tmp_path / "meta.json"
    trajectory_length = 5
    shape = [-1, 3]
    _write_meta(
        path=meta_path,
        dtypes={"a": "<dtype: 'float32'>", "b": "int32"},
        trajectory_length=trajectory_length,
        shape=shape,
    )

    sanitize_meta_json_dtypes(meta_json_path=meta_path, indent=2)

    result = json.loads(meta_path.read_text())
    assert result["features"]["a"]["dtype"] == "float32"
    assert result["features"]["b"]["dtype"] == "int32"
    assert result["trajectory_length"] == trajectory_length
    assert result["features"]["a"]["shape"] == shape


def test_sanitize_meta_json_dtypes_is_idempotent(tmp_path: Path) -> None:
    """Running the sanitizer twice produces the same file as running it once."""
    meta_path = tmp_path / "meta.json"
    _write_meta(
        path=meta_path,
        dtypes={"a": "<dtype: 'float32'>"},
        trajectory_length=5,
        shape=[-1, 3],
    )

    sanitize_meta_json_dtypes(meta_json_path=meta_path, indent=2)
    first_pass = meta_path.read_text()

    sanitize_meta_json_dtypes(meta_json_path=meta_path, indent=2)
    second_pass = meta_path.read_text()

    assert first_pass == second_pass
    assert json.loads(second_pass)["features"]["a"]["dtype"] == "float32"
