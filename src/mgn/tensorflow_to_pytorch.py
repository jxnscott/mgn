"""Loads the FlagSimple cloth dataset and caches it to disk as torch tensors."""

import functools
from pathlib import Path
from typing import Any

import tensorflow as tf
import torch
from tqdm.auto import tqdm


def _parse_protocol_buffer(
    serialized: tf.Tensor, *, metadata: dict[str, Any]
) -> dict[str, tf.Tensor]:
    """Parses one serialized trajectory record into its constituent tensors.

    Every field is stored in the tf.Example as raw bytes (via VarLenFeature(tf.string)), with its
    true dtype/shape recorded separately in `metadata`. This function decodes those bytes back into
    typed, reshaped tensors, at whatever shape `metadata` declares for them.

    Args:
        serialized: A scalar string Tensor, a single serialized Example.
        metadata: Parsed contents of the meta.json corresponding to the scalar string Tensor.

    Returns:
        A dict mapping `field_name` to its decoded Tensor.
    """
    feature_name_mapping = {
        feature_name: tf.io.VarLenFeature(tf.string) for feature_name in metadata["field_names"]
    }

    raw_byte_features = tf.io.parse_single_example(
        serialized=serialized, features=feature_name_mapping
    )

    decoded_features = {}
    for feature_name, schema in metadata["features"].items():
        out_type = tf.as_dtype(type_value=schema["dtype"])

        flat_tensors = tf.io.decode_raw(
            input_bytes=raw_byte_features[feature_name].values, out_type=out_type
        )

        decoded_features[feature_name] = tf.reshape(tensor=flat_tensors, shape=schema["shape"])

    return decoded_features


def load_tfrecord_with_metadata(
    *,
    metadata: dict[str, Any],
    tfrecord_path: Path,
    num_parallel_calls: int,
    deterministic: bool,
    buffer_size: int,
) -> tf.data.Dataset:
    """Loads a raw trajectory dataset.

    Args:
        metadata: Parsed contents of the meta.json corresponding to `tf_record_path`.
        tfrecord_path: path to the `.tfrecord` to load.
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
    lazy_dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    fused_parse_protocol_buffer = functools.partial(_parse_protocol_buffer, metadata=metadata)
    lazy_dataset = lazy_dataset.map(
        map_func=fused_parse_protocol_buffer,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )

    return lazy_dataset.prefetch(buffer_size=buffer_size)


def convert_trajectories_to_pt_and_save(*, dataset: tf.data.Dataset, out_dir: Path) -> None:
    """Converts each trajectory in ``dataset`` to torch tensors and saves it.

    Writes one ``.pt`` file per trajectory to ``out_dir``, so peak memory is
    "one trajectory at a time" rather than the whole split.

    Args:
        dataset: A dataset of parsed trajectory dicts, as returned by
            ``load_dataset``.
        out_dir: Directory to write "<index>.pt" files into. Created if it
            doesn't exist.
    """
    if out_dir.exists() and not any(out_dir.iterdir()):
        msg = f"{out_dir} is not empty. If you wish to overwrite, manually delete."
        raise RuntimeError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    for trajectory_idx, parsed_trajectory in enumerate(
        tqdm(dataset, desc="converting trajectories to .pt and saving to disk.")
    ):
        trajectory = {
            feature_name: torch.from_numpy(tf_tensor.numpy())
            for feature_name, tf_tensor in parsed_trajectory.items()
        }
        torch.save(trajectory, out_dir / f"{trajectory_idx}.pt")
