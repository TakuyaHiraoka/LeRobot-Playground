#!/usr/bin/env python
"""
hil_record_act.py  (multi-task version)
--------------------------------------
HG-DAgger style HIL recording (ACT + SO-101) / multi-task support

- By default, the loaded ACT drives the follower
- Press SPACE to toggle intervention mode -> while intervening, the leader
  arm motions are sent to the follower, and also written to the dataset
  every frame
- s : save the current episode as "success"
- r : discard the current episode and re-record
- q : end the session

If you pass multiple task strings to --tasks, a numbered selection menu
appears at the start of each episode. The selected string is used as-is
for both policy inference (task=...) and the dataset (frame["task"]).

An `is_intervention` (0/1) column is added to the saved data, so
downstream you can weight only the intervention samples / fine-tune
only on intervention spans, etc.

Prerequisites: lerobot ~0.4.x, `pip install pynput` done
"""

import argparse
import logging
import time
from contextlib import suppress

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
try:
    # lerobot >= 0.4.x: build_dataset_frame / combine_feature_dicts live here
    from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
except ImportError:
    # fallback for older / alternative layouts
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

# SO-101 config classes. The import path may change between versions.
from lerobot.robots.so_follower import SO101FollowerConfig          # noqa: E402
from lerobot.teleoperators.so_leader import SO101LeaderConfig       # noqa: E402

import sys
import termios
import tty
import select
import atexit


# ----------------------------- keyboard ---------------------------------
# pynput requires X11/Wayland, so it does not work in WSL / headless
# environments. Here we read stdin non-blockingly in cbreak mode, so
# keys are picked up only when this Python process is in the foreground
# of the terminal. The trade-off is that you cannot intervene while
# interacting with a browser or GUI (focus this terminal window before
# pressing SPACE/s/r/q).

class _StdinKeyListener:
    def __init__(self, events):
        self.events = events
        self._fd = None
        self._old = None
        self._installed = False

    def start(self):
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; keyboard input disabled. "
                  "Run this script directly in a terminal.")
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
        """Read stdin non-blockingly and update events. Call every frame."""
        if not self._installed:
            return
        # In case another library has touched termios and cleared cbreak,
        # re-apply the setting every time (workaround for cases where
        # pyserial / OpenCV etc. reset the pty).
        try:
            cur = termios.tcgetattr(self._fd)
            # Re-apply if ICANON (line input) or ECHO has come back
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
            elif ch == "r":
                self.events["rerecord"] = True
                self.events["exit_episode"] = True
            elif ch == "q":
                self.events["quit"] = True
                self.events["exit_episode"] = True
            # Ignore other keys


def make_events_and_listener():
    events = {
        "intervention": False,
        "success": False,
        "rerecord": False,
        "quit": False,
        "exit_episode": False,
    }
    listener = _StdinKeyListener(events)
    listener.start()
    if listener._installed:
        print("[HIL] keyboard listener ready "
              "(SPACE/s/r/q). keep this terminal focused.", flush=True)
    else:
        print("[HIL] keyboard listener NOT installed. "
              "Input will be ignored.", flush=True)
    return events, listener


# --------------------------- task selection -----------------------------
def choose_task(tasks, last_idx, listener=None):
    """Select a task before starting the episode. If there is only one task, return it as-is."""
    if len(tasks) == 1:
        return tasks[0], 0
    print("\n  Select task for this episode:")
    for i, t in enumerate(tasks):
        marker = "*" if i == last_idx else " "
        print(f"    {marker} [{i}] {t}")
    # input() cannot do line editing in cbreak mode, so temporarily restore it
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


# --------------------------- reset helper -------------------------------
def reset_window(robot, teleop, teleop_ap, robot_ap, duration, fps):
    """Section where the leader merely drives the follower (not written to dataset)."""
    print(f"  [reset] {duration:.1f}s  leader -> follower, not recorded")
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


# ------------------------------- main -----------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--follower_port", default="/dev/ttyACM1")
    p.add_argument("--follower_id",   default="my_awesome_follower_arm")
    p.add_argument("--leader_port",   default="/dev/ttyACM0")
    p.add_argument("--leader_id",     default="my_awesome_leader_arm")
    p.add_argument("--camera_url",    default="http://0.0.0.0:8080")
    p.add_argument("--policy_path",   required=True,
                   help="Trained ACT checkpoint (local path or HF repo_id)")
    p.add_argument("--repo_id",       required=True,
                   help="New dataset name (e.g., user/so101_pick_hil_round1)")
    p.add_argument("--root",          default=None)
    p.add_argument("--tasks",         nargs="+", required=True,
                   help="One or more task strings. If multiple are specified, a selection menu appears at the start of each episode."
                        " e.g., --tasks 'Pick Chikawa toy.' 'Pick Hachiware toy.'")
    p.add_argument("--num_episodes",  type=int,   default=20,
                   help="Number of episodes to additionally record (when resuming, this is added to the existing count).")
    p.add_argument("--episode_time_s", type=float, default=10.0)
    p.add_argument("--reset_time_s",   type=float, default=5.0)
    p.add_argument("--fps",           type=int,   default=30)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--resume",        action="store_true",
                   help="Append to an existing dataset. If not specified, create a new one.")
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

    # ---------- default pipelines (same as lerobot_record) ----------
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

    # ---------- dataset ----------
    if args.resume:
        # Open the existing dataset. features follow the existing side.
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root,
        )
        if dataset.fps != args.fps:
            raise ValueError(
                f"FPS mismatch: existing dataset fps={dataset.fps} vs --fps={args.fps}. "
                f"Use --fps={dataset.fps} to resume."
            )
        # Without starting the image writer, frame additions will stall
        if hasattr(dataset, "start_image_writer") and hasattr(robot, "cameras"):
            try:
                dataset.start_image_writer(
                    num_processes=0,
                    num_threads=4 * max(len(robot.cameras), 1),
                )
            except Exception as e:
                print(f"[WARN] start_image_writer failed ({e}); continuing.")
        has_intervention_col = "is_intervention" in dataset.features
        existing_eps = (dataset.meta.total_episodes
                        if hasattr(dataset.meta, "total_episodes")
                        else len(dataset.meta.episodes))
        print(f"[HIL] resume: opened {args.repo_id} "
              f"({existing_eps} episodes already recorded, "
              f"is_intervention column: {'yes' if has_intervention_col else 'no'})")
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

    # ---------- policy ----------
    # When a local directory is passed, normalize to an absolute path so
    # PreTrainedConfig.from_pretrained does not mistake it for an HF repo ID
    import os
    policy_path_arg = args.policy_path
    # Heuristic: if there are 2+ '/' separators but it does not match
    # the namespace/name form -> assume a local path was intended and check existence
    looks_local = ("/" in policy_path_arg
                   and policy_path_arg.count("/") != 1
                   and not policy_path_arg.startswith("hf://"))
    if looks_local:
        abspath = os.path.abspath(policy_path_arg)
        if not os.path.isdir(abspath):
            raise FileNotFoundError(
                f"policy_path appears to be a local path but does not exist:\n"
                f"  given   : {policy_path_arg}\n"
                f"  resolved: {abspath}\n"
                f"Hint: run\n"
                f"  find outputs/train -name pretrained_model -type d\n"
                f"to locate your checkpoint."
            )
        required = ["config.json"]
        missing = [f for f in required if not os.path.isfile(os.path.join(abspath, f))]
        if missing:
            raise FileNotFoundError(
                f"{abspath} exists but is missing required files: {missing}. "
                f"Make sure it is a 'pretrained_model' directory, not its parent."
            )
        policy_path_arg = abspath
        print(f"[HIL] using local policy dir: {policy_path_arg}")

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

    # ---------- keyboard listener (start early: workaround for cases where termios is touched during connect) ----------
    events, listener = make_events_and_listener()

    # ---------- connect ----------
    robot.connect()
    teleop.connect()

    recorded = 0
    last_task_idx = 0
    try:
        while recorded < args.num_episodes and not events["quit"]:
            print(f"\n=== Episode {recorded + 1}/{args.num_episodes} ===")

            # --- choose task for this episode ---
            current_task, last_task_idx = choose_task(args.tasks, last_task_idx, listener)
            print(f"  task = {current_task!r}")
            print("  SPACE: toggle intervention | s: success | r: rerecord | q: quit")

            for k in ("intervention", "success", "rerecord", "exit_episode"):
                events[k] = False

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            period = 1.0 / args.fps
            start_t = time.perf_counter()

            while True:
                loop_t0 = time.perf_counter()

                # --- poll keyboard (stdin) ---
                listener.poll()

                # --- obs ---
                obs = robot.get_observation()
                obs_proc = robot_op(obs)
                obs_frame = build_dataset_frame(dataset.features, obs_proc, prefix=OBS_STR)

                # --- always read leader (for smooth takeover) ---
                raw_teleop_act = teleop.get_action()

                # --- choose action source ---
                if events["intervention"]:
                    act_proc_teleop = teleop_ap((raw_teleop_act, obs))
                    action_values = act_proc_teleop
                    robot_action_to_send = robot_ap((act_proc_teleop, obs))
                    is_interv = 1.0
                else:
                    action_tensor = predict_action(
                        observation=obs_frame,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy_cfg.use_amp,
                        task=current_task,               # <- pass the selected task to policy
                        robot_type=robot.robot_type,
                    )
                    act_proc_policy = make_robot_action(action_tensor, dataset.features)
                    action_values = act_proc_policy
                    robot_action_to_send = robot_ap((act_proc_policy, obs))
                    is_interv = 0.0

                # --- send + record ---
                robot.send_action(robot_action_to_send)
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {
                    **obs_frame,
                    **action_frame,
                    "task": current_task,
                }
                if has_intervention_col:
                    frame["is_intervention"] = np.array([is_interv], dtype=np.float32)
                dataset.add_frame(frame)

                # --- exit conditions ---
                if events["exit_episode"]:
                    break
                if time.perf_counter() - start_t >= args.episode_time_s:
                    break

                dt = time.perf_counter() - loop_t0
                if dt < period:
                    time.sleep(period - dt)

            # --- after episode ---
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

            dataset.save_episode()
            recorded += 1
            label = "success" if events["success"] else "timeout"
            print(f"  -> saved ({label}) task={current_task!r}. total = {recorded}")

            reset_window(robot, teleop, teleop_ap, robot_ap,
                         args.reset_time_s, args.fps)

    finally:
        with suppress(Exception):
            dataset.finalize()
        with suppress(Exception):
            robot.disconnect()
        with suppress(Exception):
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        print(f"\nDone. Saved {recorded} new episodes to {dataset.root}")
        if args.push_to_hub:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
