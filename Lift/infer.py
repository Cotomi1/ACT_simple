"""Run a trained ACT policy in default Panda Lift and show the native viewer."""

import argparse
import time
from pathlib import Path

from data import ROOT


DEFAULT_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "010000" / "pretrained_model"


def load_policy(checkpoint, device):
    checkpoint = checkpoint.resolve()
    if not all((checkpoint / name).is_file() for name in ("config.json", "model.safetensors")):
        raise ValueError(f"Incomplete checkpoint at {checkpoint}; pass the pretrained_model directory")

    import draccus
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # Load the local config directly: LeRobot 0.3.3 reopens a locked temp file on Windows.
    with (checkpoint / "config.json").open(encoding="utf-8") as file, draccus.config_type("json"):
        config = draccus.load(PreTrainedConfig, file)
    if not isinstance(config, ACTConfig):
        raise ValueError(f"Expected an ACT checkpoint: {checkpoint}")
    config.device = device
    # The saved policy already includes the backbone and normalization statistics.
    config.pretrained_backbone_weights = None
    return ACTPolicy.from_pretrained(checkpoint, config=config, strict=True)


def run_episode(env, policy, max_steps):
    """Run one fresh episode and stop at the first successful lift."""
    import torch

    from environment import controller_action, policy_observation

    policy.reset()
    obs = env.reset()
    if env.has_renderer:
        env.viewer.update()
    with torch.inference_mode():
        for _ in range(max_steps):
            if env.has_renderer and not env.viewer.viewer.is_running():
                return False
            tick = time.monotonic()
            batch = {}
            for key, value in policy_observation(obs).items():
                tensor = torch.from_numpy(value)
                if key.startswith("observation.images."):
                    tensor = tensor.permute(2, 0, 1).float().div(255)
                batch[key] = tensor.unsqueeze(0).to(policy.config.device)
            action = policy.select_action(batch)
            obs, reward, done, _ = env.step(controller_action(env, action[0].cpu().numpy()))
            if reward > 0:  # Default Lift uses sparse reward.
                return True
            if done:
                break
            if env.has_renderer:
                time.sleep(max(0, env.control_timestep - (time.monotonic() - tick)))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT,
        help="pretrained_model folder (default: Lift/outputs/checkpoints/010000/pretrained_model)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    try:
        policy = load_policy(args.checkpoint, args.device)
    except ValueError as error:
        parser.error(str(error))

    from environment import make_lift

    env = make_lift(viewer=True, images=True)
    try:
        print(f"Lift success: {run_episode(env, policy, args.steps)}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
