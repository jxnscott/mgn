"""Operations related to meta.json."""

import json
import re
from pathlib import Path


def _normalize_dtype(*, dtype: str) -> str:
    """Unwraps a dtype stored as a stringified ``tf.DType`` repr.

    Some meta.json files serialize a dtype via ``str(dtype)`` instead of
    ``dtype.name``, producing e.g. "<dtype: 'float32'>" instead of
    "float32". Strips that wrapper if present; an already-plain name
    passes through unchanged.

    Args:
        dtype: The raw dtype string as read from meta.json.

    Returns:
        The plain dtype name, e.g. "float32".
    """
    match = re.compile(r"^<dtype: '(?P<name>\w+)'>$").match(dtype)
    return match["name"] if match else dtype


def sanitize_meta_json_dtypes(*, meta_json_path: Path, indent: int) -> None:
    """Rewrites meta.json in place, unwrapping any malformed dtype reprs.

    Args:
        meta_json_path: Path to the meta.json to sanitize.
        indent: Indentation width for the rewritten file.
    """
    with meta_json_path.open("r") as fp:
        metadata = json.load(fp=fp)

    for schema in metadata["features"].values():
        schema["dtype"] = _normalize_dtype(dtype=schema["dtype"])

    with meta_json_path.open("w") as fp:
        json.dump(metadata, fp, indent=indent)
