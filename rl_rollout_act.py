#!/usr/bin/env python
"""
rl_rollout_act.py
-----------------
RL rollout (with optional HIL safety net) for ACT on SO-101.

Differences from hil_record_act.py:
  * --noise_std injects Gaussian noise into the ACT latent z for exploration
    (a forward pre-hook is attached to policy.model.encoder_latent_input_proj,
     so no changes to lerobot itself are needed)
  * 'f' key explicitly marks the episode as a failure
  * On timeout, a terminal prompt always asks for success / failure /
    rerecord / quit -- every episode is guaranteed to have an outcome label
  * `next.reward` / `next.done` columns are added
      - all frames have reward = 0, done = 0
      - the last frame of a success episode has reward = 1, done = 1
      - the last frame of a failure episode has reward = 0, done = 1

For a downstream REINFORCE-style update with a binary reward and no
advantage, just keep the success episodes and re-train ACT on them with
`train_act_filtered.py` -- that is exactly the policy gradient update.

Prerequisites: lerobot ~0.4.x

Usage example:
    python rl_rollout_act.py \
        --policy_path outputs/train/act_round1/checkpoints/last/pretrained_model \
        --repo_id    user/so101_rl_round1 \
        --tasks 'Pick the red cube.' \
        --noise_std 0.3 \
        --num_episodes 50
"""

import argparse
import logging
import os
import time
from contextlib import suppress

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
try:
    from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
except ImportError:
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import init_logging, get_safe_torch_device

from lerobot.robots.so_follower import SO101FollowerConfig          # noqa: E402
from lerobot.teleoperators.so_leader import SO101LeaderConfig       # noqa: E402

import sys
import termios
import tty
import select
import atexit


# ============================ keyboard listener ===============================
class _StdinKeyListener:
    """Non-blocking key input in cbreak mode.
    SPACE = intervention / s = success / f = failure / r = rerecord / q = quit.
    """

    def __init__(self, events):
        self.events = events
        self._fd = None
        self._old = None
        self._installed = False

    def start(self):
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; keyboard input disabled.")
            return
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.stop)
        self._installed = True

    def stop(self):
        if self._installed and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._installed = False

    def poll(self):
        if not self._installed:
            return
        try:
            cur = termios.tcgetattr(self._fd)
            if cur[3] & (termios.ICANON | termios.ECHO):
                tty.setcbreak(self._fd)
        except Exception:
            pass
        while True:
            r, _, _ = select.select([self._fd], [], [], 0)
            if not r:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == " ":
                self.events["intervention"] = not self.events["intervention"]
                print(f"[HIL] intervention = {self.events['intervention']}",
                      flush=True)
            elif ch == "s":
                self.events["success"] = True
                self.events["exit_episode"] = True
            elif ch == "f":
                self.events["failure"] = True
                self.events["exit_episode"] = True
            elif ch == "r":
                self.events["rerecord"] = True
                self.events["exit_episode"] = True
            elif ch == "q":
                self.events["quit"] = True
                self.events["exit_episode"] = True


def make_events_and_listener():
    events = {
        "intervention": False,
        "success": False,
        "failure": False,
        "rerecord": False,
        "quit": False,
        "exit_episode": False,
    }
    listener = _StdinKeyListener(events)
    listener.start()
    if listener._installed:
        print("[RL] keyboard ready. SPACE: intervention | s: success | "
              "f: failure | r: rerecord | q: quit", flush=True)
    return events, listener


# ============================ ACT latent noise ================================
# At inference time, lerobot's ACT builds the latent as
#   latent_sample = torch.zeros(B, latent_dim)
# and projects it to the transformer hidden size via encoder_latent_input_proj
# (an nn.Linear). By attaching a single forward pre-hook to that Linear, we can
# replace its input with `noise * std` and explore the z space without touching
# the ACT implementation itself.

def install_latent_noise_hook(policy, noise_std: float, mode: str = "chunk"):
    """
    Attach a forward pre-hook to ACT's latent input projection.

    Args:
        policy : a loaded lerobot ACTPolicy
        noise_std : std of the additive noise. 0 disables the hook.
        mode :
          "chunk"   - fresh noise on every chunk inference (recommended)
          "episode" - reuse the same z within an episode
                      (resampled by resample_episode_noise() at episode start)

    Returns:
        state : dict for tweaking noise std / mode at runtime
        handle : torch.utils.hooks.RemovableHandle for detaching the hook
        proj_name : name of the submodule the hook was attached to (debug aid)
    """
    model = policy.model

    proj_layer = None
    proj_name = None
    for cand in ("encoder_latent_input_proj", "latent_input_proj",
                 "vae_encoder_latent_input_proj"):
        if hasattr(model, cand):
            proj_layer = getattr(model, cand)
            proj_name = cand
            break
    if proj_layer is None:
        # fallback: find a Linear whose input is latent_dim, by name
        latent_dim = getattr(policy.config, "latent_dim", None)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "latent" in n.lower():
                if latent_dim is None or m.in_features == latent_dim:
                    proj_layer = m
                    proj_name = n
                    break
    if proj_layer is None:
        raise RuntimeError(
            "Could not locate the ACT latent projection layer. "
            "Inspect model.named_modules() and add the matching name to the "
            "candidate list in install_latent_noise_hook()."
        )

    state = {
        "std": float(noise_std),
        "active": True,
        "mode": mode,           # "chunk" or "episode"
        "fixed_z": None,        # cache for episode-mode
        "latent_dim": getattr(policy.config, "latent_dim", None),
    }

    def pre_hook(module, inputs):
        if not state["active"] or state["std"] <= 0.0:
            return None
        if module.training:     # never touch the layer during training
            return None
        latent = inputs[0]      # at inference this is normally zeros
        if state["mode"] == "episode" and state["fixed_z"] is not None:
            z = state["fixed_z"].to(device=latent.device, dtype=latent.dtype)
            if z.shape[0] != latent.shape[0]:
                z = z.expand(latent.shape[0], -1).contiguous()
            return (z,)
        # default = chunk mode: fresh noise on every inference
        noise = torch.randn_like(latent) * state["std"]
        return (latent + noise,)

    handle = proj_layer.register_forward_pre_hook(pre_hook)
    print(f"[NOISE] hook installed on policy.model.{proj_name} "
          f"(std={noise_std}, mode={mode}, latent_dim={state['latent_dim']})")
    return state, handle, proj_name


def resample_episode_noise(state, device):
    """In episode-mode, draw a fresh fixed z at the start of each episode."""
    if state["mode"] != "episode" or state["std"] <= 0.0:
        return
    if state["latent_dim"] is None:
        print("[NOISE] WARN: latent_dim unknown, episode-mode noise disabled.")
        return
    state["fixed_z"] = (torch.randn(1, state["latent_dim"], device=device)
                       * state["std"])


# ============================ task selection ==================================
def choose_task(tasks, last_idx, listener=None):
    if len(tasks) == 1:
        return tasks[0], 0
    print("\n  Select task for this episode:")
    for i, t in enumerate(tasks):
        marker = "*" if i == last_idx else " "
        print(f"    {marker} [{i}] {t}")
    if listener is not None:
        listener.stop()
    try:
        raw = input(f"  number (default={last_idx}): ").strip()
    finally:
        if listener is not None:
            listener.start()
    if not raw:
        idx = last_idx
    else:
        try:
            idx = int(raw)
            if not (0 <= idx < len(tasks)):
                raise ValueError
        except ValueError:
            print("  invalid -> using default")
            idx = last_idx
    return tasks[idx], idx


def prompt_outcome_after_timeout(listener):
    """On timeout, always ask the user for success/failure."""
    if listener is not None:
        listener.stop()
    print("\n  [TIMEOUT] episode time reached. mark this episode:")
    try:
        while True:
            ans = input("    (s)uccess / (f)ailure / (r)erecord / (q)uit ? "
                        ).strip().lower()
            if ans in ("s", "success"):
                return "success"
            if ans in ("f", "failure"):
                return "failure"
            if ans in ("r", "rerecord"):
                return "rerecord"
            if ans in ("q", "quit"):
                return "quit"
            print("    invalid input.")
    finally:
        if listener is not None:
            listener.start()


# ============================ reset window ====================================
def reset_window(robot, teleop, teleop_ap, robot_ap, duration, fps):
    print(f"  [reset] {duration:.1f}s leader -> follower (not recorded)")
    period = 1.0 / fps
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        loop_t0 = time.perf_counter()
        obs = robot.get_observation()
        raw = teleop.get_action()
        act = teleop_ap((raw, obs))
        robot.send_action(robot_ap((act, obs)))
        dt = time.perf_counter() - loop_t0
        if dt < period:
            time.sleep(period - dt)


# ================================== main =====================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--follower_port", default="/dev/ttyACM1")
    p.add_argument("--follower_id",   default="my_awesome_follower_arm")
    p.add_argument("--leader_port",   default="/dev/ttyACM0")
    p.add_argument("--leader_id",     default="my_awesome_leader_arm")
    p.add_argument("--camera_url",    default="http://0.0.0.0:8080")
    p.add_argument("--policy_path",   required=True,
                   help="Base ACT checkpoint (local path or HF repo_id)")
    p.add_argument("--repo_id",       required=True,
                   help="Output dataset name (e.g., user/so101_rl_round1)")
    p.add_argument("--root",          default=None)
    p.add_argument("--tasks",         nargs="+", required=True)
    p.add_argument("--num_episodes",  type=int,   default=20)
    p.add_argument("--episode_time_s", type=float, default=10.0)
    p.add_argument("--reset_time_s",   type=float, default=5.0)
    p.add_argument("--fps",           type=int,   default=30)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--resume",        action="store_true")
    # --- RL specific ---
    p.add_argument("--noise_std",     type=float, default=0.0,
                   help="Std of Gaussian noise added to the ACT latent z "
                        "(0 = deterministic). ACT defaults are latent_dim=32 "
                        "with prior N(0, I), so 0.1 to 1.0 is the practical "
                        "range.")
    p.add_argument("--noise_mode",    choices=["chunk", "episode"],
                   default="chunk",
                   help="chunk: fresh noise on every chunk inference / "
                        "episode: reuse the same z within an episode")
    args = p.parse_args()

    init_logging()

    # ---------- robot / teleop ----------
    camera_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_url, width=640, height=480, fps=args.fps,
    )
    robot_cfg = SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras={"wrist": camera_cfg},
    )
    teleop_cfg = SO101LeaderConfig(
        port=args.leader_port, id=args.leader_id,
    )
    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_ap, robot_ap, robot_op = make_default_processors()

    # ---------- dataset features ----------
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_ap,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_op,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    dataset_features["is_intervention"] = {
        "dtype": "float32", "shape": (1,), "names": ["flag"],
    }
    dataset_features["next.reward"] = {
        "dtype": "float32", "shape": (1,), "names": ["reward"],
    }
    dataset_features["next.done"] = {
        "dtype": "float32", "shape": (1,), "names": ["done"],
    }

    # ---------- dataset ----------
    if args.resume:
        dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
        if dataset.fps != args.fps:
            raise ValueError(
                f"FPS mismatch: existing={dataset.fps} vs --fps={args.fps}.")
        if hasattr(dataset, "start_image_writer") and hasattr(robot, "cameras"):
            try:
                dataset.start_image_writer(
                    num_processes=0,
                    num_threads=4 * max(len(robot.cameras), 1),
                )
            except Exception as e:
                print(f"[WARN] start_image_writer failed ({e})")
        has_intervention_col = "is_intervention" in dataset.features
        has_reward_col = "next.reward" in dataset.features
        has_done_col = "next.done" in dataset.features
        existing = (dataset.meta.total_episodes
                    if hasattr(dataset.meta, "total_episodes")
                    else len(dataset.meta.episodes))
        print(f"[RL] resume: {args.repo_id} ({existing} eps already, "
              f"is_intervention={has_intervention_col}, "
              f"next.reward={has_reward_col}, next.done={has_done_col})")
        if not (has_reward_col and has_done_col):
            print("[RL] WARN: existing dataset is missing next.reward/done. "
                  "Either start a fresh dataset or those columns will be skipped.")
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(len(robot.cameras), 1),
        )
        has_intervention_col = True
        has_reward_col = True
        has_done_col = True

    # ---------- policy ----------
    policy_path_arg = args.policy_path
    looks_local = ("/" in policy_path_arg
                   and policy_path_arg.count("/") != 1
                   and not policy_path_arg.startswith("hf://"))
    if looks_local:
        abspath = os.path.abspath(policy_path_arg)
        if not os.path.isdir(abspath):
            raise FileNotFoundError(
                f"policy_path looks local but does not exist: {abspath}")
        policy_path_arg = abspath
        print(f"[RL] using local policy dir: {policy_path_arg}")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path_arg)
    policy_cfg.pretrained_path = policy_path_arg
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path_arg,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
        },
    )
    device = get_safe_torch_device(policy_cfg.device)

    # ---------- install latent noise hook ----------
    noise_state, noise_handle, _ = install_latent_noise_hook(
        policy, noise_std=args.noise_std, mode=args.noise_mode,
    )

    # ---------- keyboard listener ----------
    events, listener = make_events_and_listener()

    # ---------- connect ----------
    robot.connect()
    teleop.connect()

    recorded = 0
    n_success = 0
    n_failure = 0
    last_task_idx = 0
    try:
        while recorded < args.num_episodes and not events["quit"]:
            print(f"\n=== Episode {recorded + 1}/{args.num_episodes} "
                  f"(noise_std={args.noise_std}, mode={args.noise_mode}) ===")

            current_task, last_task_idx = choose_task(
                args.tasks, last_task_idx, listener)
            print(f"  task = {current_task!r}")
            print("  SPACE: intervention | s: success | f: failure | "
                  "r: rerecord | q: quit")

            for k in ("intervention", "success", "failure",
                      "rerecord", "exit_episode"):
                events[k] = False

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            # episode-mode noise: sample a fixed z at episode start
            resample_episode_noise(noise_state, device)

            period = 1.0 / args.fps
            start_t = time.perf_counter()

            while True:
                loop_t0 = time.perf_counter()

                listener.poll()

                # --- exit conditions (checked at the top so the final frame
                #     can carry the correct reward/done values) ---
                timed_out = (time.perf_counter() - start_t) >= args.episode_time_s
                user_exited = events["exit_episode"]
                is_last = timed_out or user_exited

                # if timed out and outcome is still undecided, prompt the user
                if (is_last
                        and not events["success"]
                        and not events["failure"]
                        and not events["rerecord"]
                        and not events["quit"]):
                    outcome = prompt_outcome_after_timeout(listener)
                    if outcome == "success":
                        events["success"] = True
                    elif outcome == "failure":
                        events["failure"] = True
                    elif outcome == "rerecord":
                        events["rerecord"] = True
                    elif outcome == "quit":
                        events["quit"] = True

                # rerecord/quit: bail out without appending another frame
                if events["rerecord"] or events["quit"]:
                    break

                # decide reward/done for this (last) frame
                if is_last:
                    reward_val = 1.0 if events["success"] else 0.0
                    done_val = 1.0
                else:
                    reward_val = 0.0
                    done_val = 0.0

                # --- obs ---
                obs = robot.get_observation()
                obs_proc = robot_op(obs)
                obs_frame = build_dataset_frame(
                    dataset.features, obs_proc, prefix=OBS_STR)

                # --- always read leader ---
                raw_teleop_act = teleop.get_action()

                # --- choose action source ---
                if events["intervention"]:
                    act_proc_teleop = teleop_ap((raw_teleop_act, obs))
                    action_values = act_proc_teleop
                    robot_action_to_send = robot_ap((act_proc_teleop, obs))
                    is_interv = 1.0
                else:
                    # the latent-noise hook fires inside this call
                    action_tensor = predict_action(
                        observation=obs_frame,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy_cfg.use_amp,
                        task=current_task,
                        robot_type=robot.robot_type,
                    )
                    act_proc_policy = make_robot_action(
                        action_tensor, dataset.features)
                    action_values = act_proc_policy
                    robot_action_to_send = robot_ap((act_proc_policy, obs))
                    is_interv = 0.0

                # --- send + record ---
                robot.send_action(robot_action_to_send)
                action_frame = build_dataset_frame(
                    dataset.features, action_values, prefix=ACTION)
                frame = {
                    **obs_frame,
                    **action_frame,
                    "task": current_task,
                }
                if has_intervention_col:
                    frame["is_intervention"] = np.array(
                        [is_interv], dtype=np.float32)
                if has_reward_col:
                    frame["next.reward"] = np.array(
                        [reward_val], dtype=np.float32)
                if has_done_col:
                    frame["next.done"] = np.array(
                        [done_val], dtype=np.float32)
                dataset.add_frame(frame)

                if is_last:
                    break

                dt = time.perf_counter() - loop_t0
                if dt < period:
                    time.sleep(period - dt)

            # ---- after episode ----
            if events["rerecord"]:
                print("  -> rerecord: dropping episode")
                dataset.clear_episode_buffer()
                reset_window(robot, teleop, teleop_ap, robot_ap,
                             args.reset_time_s, args.fps)
                continue
            if events["quit"]:
                print("  -> quit: dropping episode")
                dataset.clear_episode_buffer()
                break

            assert events["success"] or events["failure"], \
                "internal: episode ended without success or failure label"
            dataset.save_episode()
            recorded += 1
            if events["success"]:
                n_success += 1
                label = "SUCCESS"
            else:
                n_failure += 1
                label = "failure"
            sr = n_success / max(recorded, 1) * 100
            print(f"  -> saved [{label}] task={current_task!r}. "
                  f"total = {recorded} (success {n_success} / fail {n_failure}, "
                  f"sr = {sr:.1f}%)")

            reset_window(robot, teleop, teleop_ap, robot_ap,
                         args.reset_time_s, args.fps)

    finally:
        with suppress(Exception):
            noise_handle.remove()
        with suppress(Exception):
            dataset.finalize()
        with suppress(Exception):
            robot.disconnect()
        with suppress(Exception):
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        print(f"\nDone. saved {recorded} episodes "
              f"(success {n_success} / fail {n_failure}) to {dataset.root}")
        if args.push_to_hub:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
