from pathlib import Path
import json
import functools
from typing import Any

import tensorflow as tf

# TODO: name these better
PREFETCH = 1
NUM_CPUS = 4


def _parse_proto(proto: tf.Tensor, *, meta: dict[str, Any]) -> dict[str, tf.Tensor]:
    """Parses one serialized trajectory record into its constituent tensors.

    Every field is stored in the tf.Example as raw bytes (via
    VarLenFeature(tf.string)), with its true dtype/shape recorded
    separately in ``meta``. This decodes those bytes back into typed,
    reshaped tensors, and additionally tiles "static" fields (constant
    across the trajectory, e.g. mesh connectivity) up to a leading
    trajectory-length axis so every field has a uniform per-timestep shape.

    Args:
        proto: A scalar string tensor holding one serialized tf.Example
            record, as yielded by a TFRecordDataset.
        meta: Parsed contents of the dataset's meta.json, describing the
            dtype/shape/type of every field to decode.

    Returns:
        A dict mapping field name to its decoded tensor, each with a
        leading trajectory-length axis.

    Raises:
        ValueError: If a field's "type" in ``meta`` is not "static" or
            "dynamic".
    """
    # TODO, hand notes go through meta.json

    # Extract schemaless features
    empty_feature_container = {key: tf.io.VarLenFeature(tf.string) for key in meta["field_names"]}
    schemaless_features = tf.io.parse_single_example(proto, empty_feature_container)

    # Apply schema to features
    parsed_proto = {}
    for feature_name, schema in meta["features"].items():
        # decode the var length string tensor as the particular datatype recorded in
        # meta
        data = tf.io.decode_raw(
            schemaless_features[feature_name].values, getattr(tf, schema["dtype"])
        )

        # reshape flattened tensor into whatever shape is provided from meta
        data = tf.reshape(data, schema["shape"])

        if schema["type"] == "static":
            # data: (401, 1, 1)
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])

        elif schema["type"] != "dynamic":
            raise ValueError(f"type not static or dynamic, is: {schema['type']}")

        parsed_proto[feature_name] = data

    return parsed_proto


def load_dataset(*, path: Path, split: str) -> tf.data.Dataset:
    """Loads a raw trajectory dataset from a directory of TFRecord shards.

    Args:
        path: Directory containing "meta.json" and one "<split>.tfrecord"
            file per split.
        split: Name of the split to load, e.g. "train", "valid", or "test".
            Selects "<path>/<split>.tfrecord".

    Returns:
        A tf.data.Dataset whose elements are dicts (see ``parse_proto``) of
        decoded per-trajectory tensors, each with a leading
        trajectory-length axis.
    """
    with open(path / "meta.json", "r") as fp:
        metadata = json.loads(fp.read())

    lazy_dataset = tf.data.TFRecordDataset(str(path / f"{split}.tfrecord"))

    metadata_fused_parse = functools.partial(_parse_proto, meta=metadata)

    lazy_dataset = lazy_dataset.map(metadata_fused_parse, num_parallel_calls=8)

    # optimize performance by prefetching the next batch while the current is being
    # consumed.
    lazy_dataset = lazy_dataset.prefetch(PREFETCH)

    return lazy_dataset


def decompose_trajectories_into_training_pairs(*, dataset: tf.data.Dataset):

    def _fn(trajectory: tf.Tensor):

        output_tensor = {}
        for feature, values in trajectory.items():
            # For training we need velocity, therefore, the first training input requires
            # the previous frame so that Δx can be computed.

            output_tensor[feature] = values[1:-1]

            if feature == "world_pos":
                output_tensor["prev|" + feature] = values[0:-2]
                output_tensor["target|" + feature] = values[2:]

        return output_tensor

    return dataset.map(_fn, num_parallel_calls=NUM_CPUS)
