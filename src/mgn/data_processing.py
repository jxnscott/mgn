"""Loads the FlagSimple cloth dataset and caches it to disk as torch tensors."""

from pathlib import Path
import json
import functools
from typing import Any

import tensorflow as tf
import torch
from tqdm.auto import tqdm

PARALLEL_CALLS = 8
PREFETCH_BUFFER = 1

# Maps meta.json's string dtype names to their tf.DType. Used instead of
# getattr(tf, schema["dtype"]) — autograph's tracing of getattr with a
# dynamic string argument misbehaves in some contexts; a plain dict lookup
# sidesteps that entirely.
_DTYPE_MAP = {
    "int32": tf.int32,
    "int64": tf.int64,
    "float32": tf.float32,
    "float64": tf.float64,
}


def _parse_proto(proto: tf.Tensor, *, meta: dict[str, Any]) -> dict[str, tf.Tensor]:
    """Parses one serialized trajectory record into its constituent tensors.

    Every field is stored in the tf.Example as raw bytes (via
    VarLenFeature(tf.string)), with its true dtype/shape recorded
    separately in ``meta``. This decodes those bytes back into typed,
    reshaped tensors, at whatever shape ``meta`` declares for them —
    static fields (e.g. mesh connectivity, constant across the
    trajectory) are returned as a single frame, not tiled to a leading
    trajectory-length axis, since nothing downstream needs that
    uniformity (we don't run any TF-side per-timestep slicing).

    Args:
        proto: A scalar string tensor holding one serialized tf.Example
            record, as yielded by a TFRecordDataset.
        meta: Parsed contents of the dataset's meta.json, describing the
            dtype/shape of every field to decode.

    Returns:
        A dict mapping field name to its decoded tensor.
    """
    empty_feature_container = {key: tf.io.VarLenFeature(tf.string) for key in meta["field_names"]}
    schemaless_features = tf.io.parse_single_example(proto, empty_feature_container)

    parsed_proto = {}
    for feature_name, schema in meta["features"].items():
        data = tf.io.decode_raw(
            schemaless_features[feature_name].values, _DTYPE_MAP[schema["dtype"]]
        )
        parsed_proto[feature_name] = tf.reshape(data, schema["shape"])

    return parsed_proto


def load_dataset(*, path: Path, split: str) -> tf.data.Dataset:
    """Loads a raw trajectory dataset from a directory of TFRecord shards.

    Args:
        path: Directory containing "meta.json" and one "<split>.tfrecord"
            file per split.
        split: Name of the split to load, e.g. "train", "valid", or "test".
            Selects "<path>/<split>.tfrecord".

    Returns:
        A tf.data.Dataset whose elements are dicts (see ``_parse_proto``)
        of decoded per-trajectory tensors.
    """
    with open(path / "meta.json", "r") as fp:
        metadata = json.loads(fp.read())

    lazy_dataset = tf.data.TFRecordDataset(str(path / f"{split}.tfrecord"))

    metadata_fused_parse = functools.partial(_parse_proto, meta=metadata)
    lazy_dataset = lazy_dataset.map(
        metadata_fused_parse, num_parallel_calls=PARALLEL_CALLS, deterministic=True
    )

    # optimize performance by prefetching the next batch while the current is being
    # consumed.
    lazy_dataset = lazy_dataset.prefetch(PREFETCH_BUFFER)

    return lazy_dataset


def cache_raw_trajectories_to_disk(*, dataset: tf.data.Dataset, out_dir: Path) -> None:
    """Converts each trajectory in ``dataset`` to torch tensors and saves it.

    Writes one ``.pt`` file per trajectory to ``out_dir``, so peak memory is
    "one trajectory at a time" rather than the whole split.

    Args:
        dataset: A dataset of parsed trajectory dicts, as returned by
            ``load_dataset``.
        out_dir: Directory to write "<index>.pt" files into. Created if it
            doesn't exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, tf_tensor in enumerate(tqdm(dataset, desc="caching trajectories to disk...")):
        trajectory = {
            feature_name: torch.from_numpy(tensor.numpy())
            for feature_name, tensor in tf_tensor.items()
        }
        torch.save(trajectory, out_dir / f"{i}.pt")


def update_flag_simple_node_type_to_static(*, dir: Path) -> None:
    """Collapses each cached trajectory's ``node_type`` to a single frame.

    ``meta.json`` declares ``node_type`` as "dynamic" (stored with one
    value per timestep), but for FlagSimple it's verified constant across
    every timestep in every trajectory — possibly "dynamic" only because the schema is
    shared with FlagDynamic/SphereDynamic, which do remesh. This patches
    already-cached ``.pt`` files in place to store just one frame,
    matching how ``cells``/``mesh_pos`` are already handled.

    Args:
        dir: Directory of cached "<index>.pt" trajectory files to patch,
            as written by ``cache_raw_trajectories_to_disk``.

    Raises:
        ValueError: If ``dir`` doesn't exist or isn't a directory, or if
            any trajectory's ``node_type`` turns out not to be constant
            across time (i.e. the previously verified assumption doesn't hold).
    """
    if not dir.exists() or not dir.is_dir():
        raise ValueError(f"{dir} does not exist or is not a directory")

    for pt_file in tqdm(dir.rglob("*.pt"), desc="updating flag simple node_type to static"):
        loaded_pt = torch.load(pt_file)

        node_type = loaded_pt["node_type"]

        if not torch.equal(node_type, node_type[0].expand_as(node_type)):
            raise ValueError(
                f"{pt_file} has non-static node_type; cannot collapse to a single frame"
            )

        loaded_pt["node_type"] = loaded_pt["node_type"][0, :, :]

        torch.save(loaded_pt, pt_file)
