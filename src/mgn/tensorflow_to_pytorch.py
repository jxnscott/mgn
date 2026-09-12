"""Loads the FlagSimple cloth dataset and caches it to disk as torch tensors."""

import functools
from pathlib import Path
from typing import Any

import tensorflow as tf
import torch
from tqdm.auto import tqdm

PARALLEL_CALLS = 8
PREFETCH_BUFFER = 1


def _parse_proto(protocol_buffer: tf.Tensor, *, metadata: dict[str, Any]) -> dict[str, tf.Tensor]:
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
        protocol_buffer: A scalar string tensor holding one serialized tf.Example
            record, as yielded by a TFRecordDataset.
        metadata: Parsed contents of the meta.json corresponding to the protobuffer.

    Returns:
        A dict mapping field name to its decoded tensor.
    """
    empty_feature_container = {
        key: tf.io.VarLenFeature(tf.string) for key in metadata["field_names"]
    }
    schemaless_features = tf.io.parse_single_example(protocol_buffer, empty_feature_container)

    parsed_proto = {}
    for feature_name, schema in metadata["features"].items():
        data = tf.io.decode_raw(
            schemaless_features[feature_name].values, getattr(tf, schema["dtype"])
        )
        parsed_proto[feature_name] = tf.reshape(data, schema["shape"])

    return parsed_proto


def load_tfrecord_with_metadata(
    *,
    metadata: dict,
    tf_record_path: Path,
    num_parallel_calls: int,
    deterministic: bool,
    buffer_size: int,
) -> tf.data.Dataset:
    """Loads a raw trajectory dataset.

    Args:
        metadata: Parsed contents of the meta.json corresponding to `tf_record_path`.
        tf_record_path: path to the `.tfrecord` to load.
        num_parallel_calls: kwarg of `lazy_dataset.map()`,  how many elements get processed by
            `map_func` concurrently, instead of one at a time.
        deterministic: kwarg of `lazy_dataset.map()`, controls whether output order is preserved
            when running in parallel.
        buffer_size: kwarg of lazy_dataset.prefetch(), decouples producing dataset elements from
            consuming them by letting the pipeline prepare up to `buffer_size` elements ahead of
            time in a background thread.

    Returns:
        A tf.data.Dataset whose elements are dicts (see ``_parse_proto``)
        of decoded per-trajectory tensors.
    """
    lazy_dataset = tf.data.TFRecordDataset(str(tf_record_path))

    fused_parse_proto = functools.partial(_parse_proto, metadata=metadata)
    lazy_dataset = lazy_dataset.map(
        map_func=fused_parse_proto,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )

    # optimize performance by prefetching the next batch while the current is being
    # consumed.
    return lazy_dataset.prefetch(buffer_size=buffer_size)


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


def update_flag_simple_node_type_to_static(*, dataset_directory: Path) -> None:
    """Collapses each cached trajectory's ``node_type`` to a single frame.

    ``meta.json`` declares ``node_type`` as "dynamic" (stored with one
    value per timestep), but for FlagSimple it's verified constant across
    every timestep in every trajectory — possibly "dynamic" only because the schema is
    shared with FlagDynamic/SphereDynamic, which do remesh. This patches
    already-cached ``.pt`` files in place to store just one frame,
    matching how ``cells``/``mesh_pos`` are already handled.

    Args:
        dataset_directory: Directory of cached "<index>.pt" trajectory files to patch,
            as written by ``cache_raw_trajectories_to_disk``.

    Raises:
        ValueError: If ``dir`` doesn't exist or isn't a directory, or if
            any trajectory's ``node_type`` turns out not to be constant
            across time (i.e. the previously verified assumption doesn't hold).
    """
    if not dataset_directory.exists() or not dataset_directory.is_dir():
        msg = f"{dataset_directory} does not exist or is not a directory"
        raise ValueError(msg)

    desc = "updating flag simple node_type to static"
    for pt_file in tqdm(dataset_directory.rglob("*.pt"), desc=desc):
        loaded_pt = torch.load(pt_file)

        node_type = loaded_pt["node_type"]

        if not torch.equal(node_type, node_type[0].expand_as(node_type)):
            msg = f"{pt_file} has non-static node_type; cannot collapse to a single frame"
            raise ValueError(msg)

        loaded_pt["node_type"] = loaded_pt["node_type"][0, :, :]

        torch.save(loaded_pt, pt_file)
