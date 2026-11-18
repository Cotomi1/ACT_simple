"""Run the official LeRobot trainer with the default ACT policy."""

import argparse
import json
import logging
import sys
from pathlib import Path

from data import DATASET_ID, DATASET_PATH, ROOT


OUTPUT_PATH = ROOT / "outputs" / "act_lift"


def require_files(parser, directory, *names):
    for name in names:
        if not (directory / name).is_file():
            parser.error(f"Missing checkpoint file: {directory / name}")


def model_directory(path):
    path = path.resolve()
    if path.name == "model.safetensors":
        return path.parent
    if (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    return path


def load_resume(parser, model_dir):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.train_utils import load_training_step

    require_files(parser, model_dir, "train_config.json")
    config = TrainPipelineConfig.from_pretrained(model_dir / "train_config.json")
    state_dir = model_dir.parent / "training_state"
    require_files(parser, state_dir, "training_step.json", "rng_state.safetensors",
                  "optimizer_param_groups.json", "optimizer_state.safetensors")
    if config.scheduler is not None:
        require_files(parser, state_dir, "scheduler_state.json")
    config.resume = True
    return config, load_training_step(state_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Converted LeRobot dataset (default: data/panda_lift)")
    parser.add_argument("--output", type=Path, help="Run directory (default: outputs/act_lift)")
    parser.add_argument("--steps", type=int, help="Total optimizer updates (default for a new run: 10000)")
    parser.add_argument("--batch-size", type=int, help="Samples per update (default for a new run: 8)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Training device (default: automatic for a new run)")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path, metavar="CHECKPOINT", help="Restore a complete checkpoint")
    checkpoint.add_argument("--weights", type=Path, metavar="MODEL", help="Start a new run from local ACT weights")
    args = parser.parse_args()

    import draccus
    import torch
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import train as trainer
    from lerobot.utils.utils import init_logging

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; activate act_simple and check the NVIDIA driver and PyTorch build.")

    model_dir = None
    if checkpoint_path := args.resume or args.weights:
        model_dir = model_directory(checkpoint_path)
        require_files(parser, model_dir, "config.json", "model.safetensors")

    start_step = 0
    if args.resume:
        config, start_step = load_resume(parser, model_dir)
    else:
        config = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id=DATASET_ID),
            policy=ACTConfig(device=args.device),
            steps=10_000,
            batch_size=8,
            num_workers=0,
        )
        if args.weights:
            # Load directly: LeRobot 0.3.3 reopens a locked temporary file on Windows.
            with (model_dir / "config.json").open(encoding="utf-8") as file, draccus.config_type("json"):
                policy = draccus.load(PreTrainedConfig, file)
            policy.device = config.policy.device
            config.policy = policy
            config.policy.pretrained_path = model_dir

    if not isinstance(config.policy, ACTConfig):
        parser.error("Expected an ACT checkpoint for this project")
    if model_dir is not None:
        config.policy.pretrained_backbone_weights = None  # Model weights already include the backbone.
    if args.device is not None:
        config.policy.device = args.device
    config.policy.push_to_hub = False
    config.wandb.enable = False

    dataset = Path(args.dataset or config.dataset.root or DATASET_PATH).resolve()
    # Old checkpoints can point to the default dataset before the project was moved.
    if (
        args.resume and args.dataset is None
        and not (dataset / "meta" / "info.json").is_file()
        and dataset.parts[-2:] == DATASET_PATH.parts[-2:]
        and (DATASET_PATH / "meta" / "info.json").is_file()
    ):
        logging.warning("Using relocated dataset: %s", DATASET_PATH)
        dataset = DATASET_PATH
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        parser.error(f"Pass a converted LeRobot dataset with --dataset: {dataset}")
    if model_dir is not None:
        with info_path.open(encoding="utf-8") as file:
            features = dataset_to_policy_features(json.load(file)["features"])
        expected = config.policy.input_features | config.policy.output_features
        if features.keys() != expected.keys() or any(
            feature.type != expected[key].type or tuple(feature.shape) != tuple(expected[key].shape)
            for key, feature in features.items()
        ):
            parser.error("Dataset observation/action features do not match the ACT checkpoint")
    config.dataset.root = str(dataset)
    output = args.output
    if args.resume and output is None:
        checkpoints = model_dir.parent.parent
        if checkpoints.name != "checkpoints":
            parser.error("Pass --output when resuming a checkpoint outside its run directory")
        output = checkpoints.parent
    config.output_dir = (output or OUTPUT_PATH).resolve()
    if args.steps is not None:
        config.steps = args.steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if config.steps <= start_step:
        parser.error(f"--steps must be greater than {start_step} (the current training step)")
    if config.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.resume and (checkpoints := config.output_dir / "checkpoints").is_dir():
        if any(path.name.isdecimal() and int(path.name) > start_step for path in checkpoints.iterdir()):
            parser.error("Output already contains a later checkpoint; choose a new --output directory")

    init_logging()
    if sys.platform == "win32":
        # LeRobot 0.3.3 uses a symlink for `last`, which needs extra Windows privileges.
        trainer.update_last_checkpoint = lambda path: logging.info("Checkpoint: %s", path)
    if not args.resume:
        trainer.train(config)
        return

    logging.info("Resuming at step %d; target step %d", start_step, config.steps)
    # LeRobot's resume validation reads --config_path from argv even with a config object.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], f"--config_path={model_dir / 'train_config.json'}"]
        trainer.train(config)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
