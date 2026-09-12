"""Builds a small, truncated test fixture from a dataset's test split.

Grabs a handful of real trajectories from "<dataset>/test.tfrecord", truncates
each dynamic field down to a few frames, and writes them back out as a small
standalone .tfrecord + meta.json — real, schema-correct data, small enough to
commit as a test fixture.

Static fields (e.g. mesh connectivity) aren't truncated; they're already a
single frame in the raw encoding.

Only handles "static"/"dynamic" fields. Datasets using "dynamic_varlen"
(flag_dynamic, flag_dynamic_sizing, sphere_dynamic) are not supported.
"""

import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from mgn.data_processing import load_dataset


def _bytes_feature(raw_bytes: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_bytes]))


def truncate_trajectory(
    trajectory: dict[str, tf.Tensor], *, meta: dict[str, Any], num_frames: int
) -> dict[str, bytes]:
    """Truncates each dynamic field to ``num_frames`` and returns raw bytes per field.

    Args:
        trajectory: One parsed trajectory dict, as yielded by ``load_dataset``.
        meta: The source dataset's parsed meta.json, describing which fields
            are "static" vs "dynamic".
        num_frames: How many leading frames to keep for dynamic fields.

    Returns:
        A dict mapping field name to its (possibly truncated) raw bytes,
        ready to be wrapped in a tf.Example.
    """
    raw_fields = {}
    for feature_name, schema in meta["features"].items():
        if schema["type"] not in ("static", "dynamic"):
            msg = (
                f"{feature_name} has unsupported type {schema['type']!r}; "
                "only 'static'/'dynamic' fields are handled"
            )
            raise ValueError(msg)
        tensor = trajectory[feature_name]
        if schema["type"] == "dynamic":
            tensor = tensor[:num_frames]
        raw_fields[feature_name] = tensor.numpy().tobytes()
    return raw_fields


def build_truncated_meta(meta: dict[str, Any], *, num_frames: int) -> dict[str, Any]:
    """Returns a copy of ``meta`` with dynamic fields sized to ``num_frames``.

    Args:
        meta: The source dataset's parsed meta.json.
        num_frames: The truncated frame count to record.

    Returns:
        A deep copy of ``meta`` with every dynamic field's ``shape[0]`` and
        the top-level ``trajectory_length`` set to ``num_frames``.
    """
    new_meta = json.loads(json.dumps(meta))
    new_meta["trajectory_length"] = num_frames
    for feature_name, schema in new_meta["features"].items():
        if schema["type"] not in ("static", "dynamic"):
            msg = (
                f"{feature_name} has unsupported type {schema['type']!r}; "
                "only 'static'/'dynamic' fields are handled"
            )
            raise ValueError(msg)
        if schema["type"] == "dynamic":
            schema["shape"][0] = num_frames
    return new_meta


def main(
    *,
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    num_trajectories: int,
    num_frames: int,
) -> None:
    """Builds a truncated test fixture for one dataset.

    Args:
        dataset_name: Name of the dataset subdirectory, e.g. "flag_simple".
        data_dir: Directory containing "<dataset_name>/test.tfrecord" and
            "<dataset_name>/meta.json".
        output_dir: Directory to write the truncated "test.tfrecord" and
            "meta.json" into. Created if it doesn't exist.
        num_trajectories: How many trajectories to pull from the source
            test split.
        num_frames: How many leading frames to keep per trajectory, for
            dynamic fields.
    """
    dataset_dir = data_dir / dataset_name

    with (dataset_dir / "meta.json").open(mode="r") as fp:
        meta = json.loads(fp.read())

    ds = load_dataset(path=dataset_dir, split="test")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tf.io.TFRecordWriter(str(output_dir / "test.tfrecord")) as writer:
        for trajectory in ds.take(num_trajectories):
            raw_fields = truncate_trajectory(trajectory, meta=meta, num_frames=num_frames)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={name: _bytes_feature(raw) for name, raw in raw_fields.items()}
                )
            )
            writer.write(example.SerializeToString())

    truncated_meta = build_truncated_meta(meta, num_frames=num_frames)
    with (output_dir / "meta.json").open(mode="w") as fp:
        json.dump(truncated_meta, fp, indent=2)

    print(
        f"Wrote {num_trajectories} truncated trajectories ({num_frames} frames each) "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    DATASET_NAME = "flag_dynamic_sizing"
    DATA_DIR = Path("/mnt/drives/samsung_ssd/mgn/data/")
    OUTPUT_DIR = Path(f"/home/jxn/dev/meshgraphnets/mgn/tests/data/{DATASET_NAME}")
    NUM_TRAJECTORIES = 2
    NUM_FRAMES = 10

    main(
        dataset_name=DATASET_NAME,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        num_trajectories=NUM_TRAJECTORIES,
        num_frames=NUM_FRAMES,
    )
