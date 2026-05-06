#!/usr/bin/env python
"""
train_act_filtered.py
---------------------
REINFORCE-style ACT re-training (binary reward, no advantage).

For a binary reward R in {0, 1} and no advantage, the REINFORCE update is

    grad_theta J  =  E_tau [ grad_theta log pi(tau) * R(tau) ]

  - R = 0 episodes contribute no gradient -> drop them.
  - R = 1 episodes pull log pi up         -> just do MLE on those rollouts.

So "keep the success episodes and re-train ACT on them" *is* the REINFORCE
update. This script does that in two stages:

  1. Open the dataset, look at each episode's max(next.reward), and build a
     list of success episode indices.
  2. Spawn `lerobot-train` as a subprocess with `--dataset.episodes=[...]`
     so only the success episodes are seen during training.

The actual training is fully delegated to lerobot-train, so the optimizer,
LR schedule, logging, and checkpointing all follow lerobot's conventions.

Usage:
    python train_act_filtered.py \
        --dataset_repo_id user/so101_rl_round1 \
        --dataset_root    /path/to/dataset \
        --base_policy     outputs/train/act_round0/checkpoints/last/pretrained_model \
        --output_dir      outputs/train/act_round1 \
        -- \
        --steps 30000 \
        --batch_size 64 \
        --policy.optimizer_lr 1e-5

Anything after `--` is forwarded verbatim to lerobot-train.
"""

import argparse
import json
import subprocess
import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# --------------------------- success episode finder --------------------------
def find_success_episodes(dataset, threshold: float = 0.5):
    """
    For each episode, mark it as a success if max(next.reward) > threshold.
    Returns the list of success episode indices.
    """
    n_eps = dataset.meta.total_episodes
    if n_eps == 0:
        return []

    hf = dataset.hf_dataset

    # Pull columns out as numpy and scan in O(N). filter() is too slow.
    try:
        ep_idx_col = np.asarray(hf["episode_index"])
        reward_raw = hf["next.reward"]
        # next.reward is stored with shape (1,), so the column tends to be 2D.
        reward_arr = np.asarray(reward_raw)
        if reward_arr.dtype == object:
            # object array (each element is a list/array) -> coerce to float
            reward_arr = np.array(
                [float(r[0] if hasattr(r, "__len__") else r)
                 for r in reward_raw],
                dtype=np.float32,
            )
        elif reward_arr.ndim == 2 and reward_arr.shape[1] == 1:
            reward_arr = reward_arr[:, 0]
        reward_arr = reward_arr.astype(np.float32)
    except Exception as e:
        # defensive fallback for older lerobot or non-standard storage
        print(f"[FILTER] direct column read failed ({e}); "
              f"falling back to per-row iteration.")
        ep_idx_col = []
        reward_arr = []
        for row in hf:
            ep_idx_col.append(int(row["episode_index"]))
            r = row["next.reward"]
            reward_arr.append(
                float(r[0]) if hasattr(r, "__len__") else float(r))
        ep_idx_col = np.asarray(ep_idx_col)
        reward_arr = np.asarray(reward_arr, dtype=np.float32)

    success_eps = []
    for ep_idx in range(n_eps):
        mask = (ep_idx_col == ep_idx)
        if not mask.any():
            continue
        if float(reward_arr[mask].max()) > threshold:
            success_eps.append(ep_idx)
    return success_eps


# --------------------------------- main --------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset_repo_id", required=True,
                   help="The rollout dataset to filter.")
    p.add_argument("--dataset_root", default=None,
                   help="Local dataset root (optional).")
    p.add_argument("--base_policy", required=True,
                   help="Path to the ACT checkpoint to fine-tune from.")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for lerobot-train.")
    p.add_argument("--reward_threshold", type=float, default=0.5,
                   help="An episode counts as a success when "
                        "max(next.reward) is greater than this (default 0.5).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print success/failure stats only; don't train.")
    p.add_argument("--train_cmd", default="lerobot-train",
                   help="Training command name (default: lerobot-train). "
                        "If that is not on PATH, try "
                        "'python -m lerobot.scripts.train' etc.")
    p.add_argument("--print_failures", action="store_true",
                   help="Also print the failure episode indices.")
    args, extra_args = p.parse_known_args()

    # Treat `--` as a separator: anything after it is forwarded verbatim.
    # parse_known_args() leaves unconsumed arguments in extra_args.
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # ---------- open dataset ----------
    print(f"[FILTER] opening dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id, root=args.dataset_root)
    n_eps = dataset.meta.total_episodes
    print(f"[FILTER] total episodes: {n_eps}")

    if "next.reward" not in dataset.features:
        print("[FILTER] ERROR: dataset has no 'next.reward' column. "
              "Use a dataset recorded with rl_rollout_act.py.")
        sys.exit(2)

    # ---------- find success episodes ----------
    success_eps = find_success_episodes(
        dataset, threshold=args.reward_threshold)
    failure_eps = [i for i in range(n_eps) if i not in success_eps]

    sr = len(success_eps) / max(n_eps, 1) * 100
    print(f"[FILTER] success: {len(success_eps)}/{n_eps}  "
          f"(success rate {sr:.1f}%)")
    if len(success_eps) <= 50:
        print(f"  success indices: {success_eps}")
    else:
        print(f"  success indices (head): {success_eps[:50]} ...")
    if args.print_failures:
        if len(failure_eps) <= 50:
            print(f"  failure indices: {failure_eps}")
        else:
            print(f"  failure indices (head): {failure_eps[:50]} ...")

    if len(success_eps) == 0:
        print("[FILTER] ERROR: no success episodes -> nothing to train on. "
              "Increase exploration noise, collect more demos, or revisit the "
              "reward signal.")
        sys.exit(1)

    if args.dry_run:
        print("[FILTER] --dry_run: skipping training.")
        return

    # ---------- build & launch lerobot-train ----------
    # The lerobot CLI accepts a python literal list here.
    ep_list_str = "[" + ",".join(str(e) for e in success_eps) + "]"

    cmd = list(args.train_cmd.split()) + [
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.episodes={ep_list_str}",
        f"--policy.path={args.base_policy}",
        f"--output_dir={args.output_dir}",
    ]
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    cmd.extend(extra_args)

    print("\n[TRAIN] invoking:")
    for i, c in enumerate(cmd):
        sep = "    " if i > 0 else "  "
        cont = " \\" if i < len(cmd) - 1 else ""
        print(f"{sep}{c}{cont}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[TRAIN] lerobot-train exited with code {result.returncode}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
