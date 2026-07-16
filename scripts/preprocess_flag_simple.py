import argparse
from pathlib import Path

from mgn.data_processing import (
    load_dataset,
    cache_raw_trajectories_to_disk,
    update_flag_simple_node_type_to_static,
)
from mgn.seed import seed_everything

SEED = 0


def main(*, dataset_dir: Path) -> None:
    # TODO: Optimize and clean this up.

    seed_everything(seed=SEED)

    # 1. Cache Everything to Disk
    splits = ["train", "valid", "test"]
    for split in splits:
        ds = load_dataset(path=dataset_dir, split=split)

        out_dir = dataset_dir / "pytorch" / split

        cache_raw_trajectories_to_disk(dataset=ds, out_dir=out_dir)
        update_flag_simple_node_type_to_static(dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache the FlagSimple dataset to disk as torch tensors."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the raw flag_simple .tfrecord files and meta.json.",
    )
    args = parser.parse_args()

    main(dataset_dir=args.dataset_dir)
