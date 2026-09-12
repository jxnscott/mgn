"""simple test."""

import json
from pathlib import Path

from mgn.meta import sanitize_meta_json_dtypes
from mgn.tensorflow_to_pytorch import load_tfrecord_with_metadata

tfrecord_path = Path("/mnt/drives/samsung_ssd/mgn/data/flag_dynamic_sizing/test.tfrecord")
meta_json_path = Path("/mnt/drives/samsung_ssd/mgn/data/flag_dynamic_sizing/meta.json")

sanitize_meta_json_dtypes(meta_json_path=meta_json_path, indent=4)

with meta_json_path.open(mode="r") as fp:
    metadata = json.load(fp)


dataset = load_tfrecord_with_metadata(
    metadata=metadata,
    tfrecord_path=tfrecord_path,
    deterministic=True,
    num_parallel_calls=8,
    buffer_size=1,
)

first_trajectory = next(iter(dataset))

print(first_trajectory)
