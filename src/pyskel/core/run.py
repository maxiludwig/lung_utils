"""Runner which executes the main routine of LungUtils."""

import logging
import time
from typing import Any

from lung_utils.core.example import exemplary_function
from lung_utils.core.utilities import RunManager

log = logging.getLogger("lung_utils")


def run_lung_utils(config: Any) -> None:
    """General run procedure of LungUtils.

    Args:
        config (Any): Munch type object containing all configs for current
        run. Config options can be called via attribute-style access.
    """

    # Time
    start_time = time.time()

    # Run manager to handle overall tasks
    run_manager = RunManager(config)
    run_manager.init_run()

    # add overall execution here
    log.info("Output of exemplary program: " + str(exemplary_function(2, 4)))

    # finalize run
    run_manager.finish_run(start_time)
