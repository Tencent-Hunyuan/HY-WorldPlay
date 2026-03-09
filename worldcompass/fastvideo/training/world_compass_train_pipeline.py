# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import os
import sys

sys.path.append(os.path.abspath("."))

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.world_compass_in_train_pipeline import (
    WorldCompassInTrainPipeline,
)
from fastvideo.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WorldCompassTrainPipeline(WorldCompassInTrainPipeline):
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WorldCompassTrainPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
