"""Convert the official robomimic Lift PH states to a local LeRobot dataset."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_REPO = "robomimic/robomimic_datasets"
SOURCE_FILE = "v1.5/lift/ph/demo_v15.hdf5"
RAW_PATH = ROOT / "data" / "demo_v15.hdf5"
DATASET_PATH = ROOT / "data" / "panda_lift"
DATASET_ID = "local/panda_lift"
TASK = "Lift the cube off the table."


def download_source():
    from huggingface_hub import hf_hub_download

    return RAW_PATH if RAW_PATH.is_file() else Path(
        hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset")
    )


def convert(source, output, limit=None):
    import h5py
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    from environment import ACTION_NAMES, CAMERAS, IMAGE_SIZE, STATE_NAMES, make_lift, policy_observation

    if output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {output}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        **{
            f"observation.images.{camera}": {
                "dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERAS
        },
    }

    with h5py.File(source, "r") as file:
        demos = file["data"]
        meta = json.loads(demos.attrs["env_args"])
        settings = meta["env_kwargs"]
        arm = settings["controller_configs"]["body_parts"]["right"]
        if (
            meta["env_name"], settings["robots"], settings["control_freq"],
            arm["type"], arm["input_ref_frame"], arm["control_delta"],
        ) != ("Lift", ["Panda"], 20, "OSC_POSE", "world", True):
            raise ValueError("Expected the official v1.5 20 Hz Panda Lift world-delta dataset.")
        keys = sorted(demos, key=lambda key: int(key.rsplit("_", 1)[1]))[:limit]
        env = make_lift(images=True)
        try:
            dataset = LeRobotDataset.create(
                repo_id=DATASET_ID, root=output, fps=env.control_freq,
                robot_type="panda", features=features, use_videos=False,
            )
            for key in tqdm(keys, desc="Panda Lift episodes"):
                demo = demos[key]
                env.reset_from_xml_string(demo.attrs["model_file"])
                # State and action share index t. Render state t directly to avoid replay drift.
                for state, action in zip(demo["states"], demo["actions"], strict=True):
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    frame = policy_observation(env._get_observations(force_update=True))
                    frame["action"] = np.asarray(action, dtype=np.float32)
                    dataset.add_frame(frame, task=TASK)
                dataset.save_episode()
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local robomimic HDF5 file")
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--episodes", type=int, help="Convert only the first N episodes for a smoke check")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    convert(args.source or download_source(), args.output.resolve(), args.episodes)


if __name__ == "__main__":
    main()
