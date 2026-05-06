# SO-101 ACT: HIL + RL Fine-tuning Pipeline

A small set of scripts that wrap [LeRobot](https://github.com/huggingface/lerobot) to fine-tune an ACT policy on the SO-101 arm pair in a closed loop:

1. Train an initial ACT policy from teleop demos (upstream `lerobot-train`).
2. Grow the dataset with human-in-the-loop corrections — HG-DAgger style — using `hil_record_act.py`.
3. Roll out the policy with latent-space noise to collect labeled success / failure episodes using `rl_rollout_act.py`.
4. Either:
   - re-train ACT on the success episodes only with `train_act_filtered.py` — equivalent to a REINFORCE update for binary reward, or
   - train a binary success classifier as a learned value function with `train_reward_classifier.py`, and visualize its predictions over episode rollouts with `visualize_value_predictions.py`.

## Pipeline

```
   demos
     │
     ▼
lerobot-train ──────► ACT v0
                        │
                        ▼
              hil_record_act.py  (HG-DAgger HIL data collection)
                        │
                        ▼
                     dataset
                        │
                        ▼
              lerobot-train ───► ACT v1
                                   │
                                   ▼
                       rl_rollout_act.py  (rollouts + outcome labels)
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
         train_act_filtered.py        train_reward_classifier.py
                  │                                 │
                  ▼                                 ▼
               ACT v2                          value model
                                                   │
                                                   ▼
                                  visualize_value_predictions.py
```

## Requirements

- Python 3.10 (tested with 3.10.19)
- `lerobot == 0.4.4`
- SO-101 leader and follower arms (or another two-arm setup supported by LeRobot)
- A camera (USB or IP / MJPEG stream)
- A Linux terminal — the keyboard listener uses `termios` / `tty` and will not work on Windows

```bash
pip install lerobot==0.4.4
```

## Scripts

### `hil_record_act.py` — HG-DAgger style HIL recording

Loads a trained ACT policy and lets it drive the follower. While recording, you can take over by pressing **SPACE** to toggle intervention mode; in that mode the leader arm drives the follower and every frame is logged with `is_intervention = 1`.

Keys (terminal must be focused):

| key     | action                                  |
| ------- | --------------------------------------- |
| `SPACE` | toggle intervention                     |
| `s`     | save current episode as success         |
| `r`     | discard and re-record                   |
| `q`     | end the session                         |

```bash
python hil_record_act.py \
    --follower_port /dev/ttyACM1 --follower_id <follower-id> \
    --leader_port   /dev/ttyACM0 --leader_id   <leader-id> \
    --camera_url    "http://<camera-host>:8080" \
    --policy_path   ./outputs/train/<act-run-name>/checkpoints/<step>/pretrained_model \
    --repo_id       <your-hf-username>/<dataset-name> \
    --tasks         "Pick the red cube." \
    --num_episodes 20 --episode_time_s 10 --reset_time_s 5 --fps 30
```

Multiple task strings can be passed to `--tasks`; a numbered selection menu appears at the start of each episode.


Data collected with hil_record_act.py: https://huggingface.co/datasets/TakuyaHiraoka/so101_pick_diverse_objects_hil_round1  
Policy trained on the above data: https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects

https://github.com/user-attachments/assets/b4571de0-7ce4-4a43-b80f-ef2babf4460b


### `rl_rollout_act.py` — exploration rollouts with latent noise + outcome labels

Same control flow as `hil_record_act.py`, but adds:

- `--noise_std` injects Gaussian noise into the ACT latent `z` for exploration. Implemented as a forward pre-hook on the latent input projection — no edits to LeRobot itself.
- `f` explicitly marks an episode as a failure.
- On timeout, a terminal prompt always asks for success / failure / re-record / quit, so every saved episode has an outcome label.
- `next.reward` and `next.done` columns are added: `0 / 0` for all frames, with the last frame of a success episode getting `reward = 1, done = 1`.

```bash
python rl_rollout_act.py \
    --follower_port /dev/ttyACM1 --follower_id <follower-id> \
    --leader_port   /dev/ttyACM0 --leader_id   <leader-id> \
    --camera_url    "http://<camera-host>:8080" \
    --policy_path   ./outputs/train/<act-run-name>/checkpoints/<step>/pretrained_model \
    --repo_id       <your-hf-username>/<dataset-name> \
    --tasks         "Pick a pen." \
    --num_episodes 20 --episode_time_s 20 --reset_time_s 5 --fps 30 \
    --noise_std 0.05 --resume
```

### `train_act_filtered.py` — REINFORCE-style ACT re-training

For binary reward `R ∈ {0, 1}` and no advantage, the REINFORCE update collapses to standard supervised learning on the success episodes only:

```
∇θ J = E_τ [ ∇θ log π(τ) · R(τ) ]
```

The script scans the dataset for `max(next.reward) > threshold`, builds the list of success episode indices, and forwards them to `lerobot-train` via `--dataset.episodes=[...]`. Anything after `--` is passed verbatim to `lerobot-train`.

```bash
python train_act_filtered.py \
    --dataset_repo_id <your-hf-username>/<dataset-name> \
    --base_policy     ./outputs/train/<act-run-name>/checkpoints/<step>/pretrained_model \
    --output_dir      outputs/train/<next-act-run-name> \
    -- \
    --steps 30000 --batch_size 64 --policy.optimizer_lr 1e-5
```

Use `--dry_run` to print the success / failure breakdown without launching training.

### `train_reward_classifier.py` — binary success classifier (value head)

Trains a ResNet-18 backbone with a single-logit head on per-frame Monte-Carlo labels: every frame of a success episode is labeled `1`, every frame of a failure episode is labeled `0`. The output approximates `P(success | s)` — a value function for a binary terminal reward.

```bash
python train_reward_classifier.py \
    --dataset_repo_id <your-hf-username>/<dataset-name> \
    --output_dir      outputs/reward_clf/round1 \
    --epochs 20 --batch_size 64
```

Validation episodes are held out at the *episode* level (not the frame level) to avoid leakage.

### `visualize_value_predictions.py` — overlay value predictions on episode videos

Replays a few success and failure episodes and overlays the trained classifier's per-frame predictions on the wrist-camera footage. Useful for sanity-checking what the value head is actually keying on.

```bash
python visualize_value_predictions.py \
    --dataset_repo_id <your-hf-username>/<dataset-name> \
    --classifier_ckpt outputs/reward_clf/round1/best.pt \
    --output_dir      outputs/reward_clf/round1/videos \
    --num_success 3 --num_failure 3
```

Demo video:



https://github.com/user-attachments/assets/c083de6f-5549-4d96-af49-5b36c85d2c45







## Notes

- All scripts use `tty.setcbreak` on stdin for non-blocking key input, so keep the terminal window focused while recording.
- `--camera_url` must be an address the OpenCV `VideoCapture` backend can actually dial. If the camera is on the same host, prefer `http://127.0.0.1:8080` over `0.0.0.0`.
- `--push_to_hub` will push the recorded dataset to your Hugging Face account; make sure you are authenticated first (`huggingface-cli login`).
- `<step>` placeholders refer to LeRobot training-step subdirectories (e.g. `checkpoints/100000/pretrained_model`); use `last` to point at the most recent one.

