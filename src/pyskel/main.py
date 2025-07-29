"""Main routine of LungUtils."""

import argparse
import os

import yaml
from munch import munchify
from lung_utils.core.run import run_lung_utils


def main() -> None:
    """Call LungUtils runner with config.

    Raises:
        RuntimeError: If provided config is not a valid file.
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config_file_path",
        "-cfp",
        help="Path to config file.",
        type=str,
        default="src/lung_utils/main_example_config.yaml",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_file_path):
        raise RuntimeError(
            "Config file not found! LungUtils can not be executed!"
        )

    # load config and convert to simple namespace for easier access
    with open(args.config_file_path, "r") as file:
        config = munchify(yaml.safe_load(file))

    # execute lung_utils
    run_lung_utils(config)


if __name__ == "__main__":  # pragma: no cover

    main()
    exit(0)
