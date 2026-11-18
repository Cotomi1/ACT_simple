"""Evaluate a Lift checkpoint and plot success/failure counts."""

import argparse
import sys
from pathlib import Path

VISUALIZATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUALIZATION.parent / "Lift"))

from infer import DEFAULT_CHECKPOINT, load_policy, run_episode


def save_plot(path, successes, episodes, seed, steps):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(5.5, 3.6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    bars = ax.bar(["Success", "Failure"], [successes, episodes - successes],
                  width=0.55, color=["#217a72", "#bd5544"])
    ax.bar_label(bars, padding=4, fontsize=12)
    ax.set(title=f"Panda Lift: {successes}/{episodes} successful ({successes / episodes:.0%})",
           ylabel="Episodes", ylim=(0, episodes * 1.15 + 1))
    ax.set_xlabel(f"Seed {seed} | up to {steps} steps per episode")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=VISUALIZATION / "Plots" / "lift_success.png")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.seed < 0:
        parser.error("episodes and steps must be positive; seed must be nonnegative")

    from tqdm import trange

    from environment import make_lift

    try:
        policy = load_policy(args.checkpoint, args.device)
    except ValueError as error:
        parser.error(str(error))

    # Keep both policy cameras active without opening 50 viewer windows.
    env = make_lift(images=True, seed=args.seed)
    try:
        successes = sum(run_episode(env, policy, args.steps)
                        for _ in trange(args.episodes, desc="Panda Lift"))
    finally:
        env.close()

    save_plot(args.output, successes, args.episodes, args.seed, args.steps)
    print(f"Success: {successes}, failure: {args.episodes - successes} | {args.output.resolve()}")


if __name__ == "__main__":
    main()
