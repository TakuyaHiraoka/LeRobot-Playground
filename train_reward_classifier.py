#!/usr/bin/env python
"""
train_reward_classifier.py
--------------------------
Train a binary success/failure classifier (a value function in the Monte
Carlo return sense) on a rollout dataset recorded with rl_rollout_act.py.

Per-frame label = 1 if the episode this frame belongs to ended in success
(max(next.reward) > threshold), else 0. This is Monte Carlo return labeling
for a binary terminal reward, so V_phi(s) approximates P(success | s).

Architecture:
  ResNet18 backbone (ImageNet-pretrained) -> avgpool -> Linear(512, 1).
  Image-only by default. State input can be added by concatenating onto
  the pooled feature.

Usage:
    python train_reward_classifier.py \
        --dataset_repo_id user/so101_rl_round1 \
        --dataset_root    /path/to/dataset \
        --output_dir      outputs/reward_clf/round1 \
        --image_key       observation.images.wrist \
        --epochs 30 --batch_size 64 --lr 1e-4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================ label construction ============================
def build_frame_labels(dataset: LeRobotDataset,
                       reward_threshold: float = 0.5,
                       positive_window_frames: int = 0):
    """
    Build per-(global frame) label tensor.

    Args:
        dataset : opened LeRobotDataset
        reward_threshold : episode is success if max(next.reward) > threshold
        positive_window_frames :
          0  -> label every frame in a success episode as 1 (Monte Carlo)
          K>0 -> label only the last K frames of a success episode as 1,
                 earlier success frames are dropped from training (label = -1)
                 and failure frames remain 0
    Returns:
        global_indices : np.int64 array of frame indices to use
        labels         : np.float32 array of {0, 1}
        per_episode    : dict { ep_idx -> "success" | "failure" }
    """
    hf = dataset.hf_dataset
    n_eps = dataset.meta.total_episodes

    ep_idx_col = np.asarray(hf["episode_index"], dtype=np.int64)
    frame_idx_col = np.asarray(hf["frame_index"], dtype=np.int64)

    reward_raw = hf["next.reward"]
    reward_arr = np.asarray(reward_raw)
    if reward_arr.dtype == object:
        reward_arr = np.array(
            [float(r[0] if hasattr(r, "__len__") else r) for r in reward_raw],
            dtype=np.float32,
        )
    elif reward_arr.ndim == 2 and reward_arr.shape[1] == 1:
        reward_arr = reward_arr[:, 0]
    reward_arr = reward_arr.astype(np.float32)

    per_episode = {}
    chosen_indices = []
    chosen_labels = []

    for ep in range(n_eps):
        mask = (ep_idx_col == ep)
        if not mask.any():
            continue
        global_idx_in_ep = np.where(mask)[0]      # global frame indices
        # sort by frame_index just in case order is not monotonic
        order = np.argsort(frame_idx_col[global_idx_in_ep])
        global_idx_in_ep = global_idx_in_ep[order]

        max_r = float(reward_arr[mask].max())
        is_success = max_r > reward_threshold
        per_episode[ep] = "success" if is_success else "failure"

        if is_success:
            if positive_window_frames > 0:
                pos = global_idx_in_ep[-positive_window_frames:]
                # earlier success frames are simply unused (no negative
                # labeling, since they actually came from a success episode)
                chosen_indices.append(pos)
                chosen_labels.append(np.ones(len(pos), dtype=np.float32))
            else:
                chosen_indices.append(global_idx_in_ep)
                chosen_labels.append(
                    np.ones(len(global_idx_in_ep), dtype=np.float32))
        else:
            # all frames of failure episodes count as negatives
            chosen_indices.append(global_idx_in_ep)
            chosen_labels.append(
                np.zeros(len(global_idx_in_ep), dtype=np.float32))

    global_indices = np.concatenate(chosen_indices) if chosen_indices \
        else np.zeros(0, dtype=np.int64)
    labels = np.concatenate(chosen_labels) if chosen_labels \
        else np.zeros(0, dtype=np.float32)
    return global_indices, labels, per_episode


# ============================ pytorch dataset ===============================
class FrameClassifierDataset(Dataset):
    """
    Wraps a LeRobotDataset to yield (image_tensor, label) for a chosen subset
    of global frame indices. The image transform is applied here, so this
    layer is independent of any policy preprocessing.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, lerobot_ds, global_indices, labels,
                 image_key, image_size=224, train=True):
        self.ds = lerobot_ds
        self.indices = np.asarray(global_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.image_key = image_key

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        item = self.ds[gi]
        img = item[self.image_key]                         # (C, H, W) float [0,1]
        if img.ndim == 4:                                  # (T, C, H, W)
            img = img[-1]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if img.dtype != torch.float32:
            img = img.float()
        # if it slipped through as 0..255, normalize
        if img.max() > 1.5:
            img = img / 255.0
        img = self.transform(img)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return img, label


# ================================ model ====================================
class ResNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 \
            if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)   # (B,) raw logit


# ============================== train loop =================================
def split_indices_by_episode(global_indices, hf_dataset, val_fraction, seed):
    """Hold out whole episodes for validation to avoid leakage."""
    rng = np.random.default_rng(seed)
    ep_idx_col = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    eps = np.unique(ep_idx_col[global_indices])
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_fraction))) if len(eps) >= 4 else 1
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([ep_idx_col[gi] in val_eps for gi in global_indices])
    return ~is_val, is_val, sorted(val_eps)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0; loss_sum = 0.0; correct = 0
    for img, y in loader:
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(img)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = (logits.sigmoid() > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    total = 0; loss_sum = 0.0; correct = 0
    tp = fp = tn = fn = 0
    for img, y in loader:
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(img)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        pred = (logits.sigmoid() > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.numel()
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return avg_loss, acc, prec, rec


# ================================== main ===================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_key", default=None,
                   help="e.g. observation.images.wrist. Auto-detected if "
                        "omitted.")
    p.add_argument("--reward_threshold", type=float, default=0.5)
    p.add_argument("--positive_window_frames", type=int, default=0,
                   help="0 = Monte-Carlo label all success-episode frames "
                        "as 1; K>0 = only the last K frames of each success "
                        "episode are positive.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_pretrained", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- dataset ----------
    print(f"[CLF] opening dataset: {args.dataset_repo_id}")
    ds = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
    print(f"[CLF] total episodes: {ds.meta.total_episodes}")

    if "next.reward" not in ds.features:
        print("[CLF] ERROR: dataset has no 'next.reward' column.")
        sys.exit(2)

    image_keys = [k for k in ds.features
                  if k.startswith("observation.images.")]
    if not image_keys:
        print("[CLF] ERROR: no observation.images.* keys in dataset.")
        sys.exit(2)
    if args.image_key is None:
        args.image_key = image_keys[0]
        print(f"[CLF] auto-detected image_key = {args.image_key} "
              f"(available: {image_keys})")
    elif args.image_key not in image_keys:
        print(f"[CLF] ERROR: --image_key {args.image_key} not found. "
              f"available: {image_keys}")
        sys.exit(2)

    # ---------- labels ----------
    indices, labels, per_ep = build_frame_labels(
        ds,
        reward_threshold=args.reward_threshold,
        positive_window_frames=args.positive_window_frames,
    )
    n_succ_eps = sum(1 for v in per_ep.values() if v == "success")
    n_fail_eps = sum(1 for v in per_ep.values() if v == "failure")
    print(f"[CLF] success eps {n_succ_eps} / failure eps {n_fail_eps}")
    print(f"[CLF] training frames: {len(indices)}  "
          f"(pos {int(labels.sum())} / neg {int((1 - labels).sum())})")
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        print("[CLF] ERROR: need at least one success and one failure "
              "episode to train.")
        sys.exit(1)

    # ---------- split ----------
    train_mask, val_mask, val_eps = split_indices_by_episode(
        indices, ds.hf_dataset, args.val_fraction, args.seed)
    print(f"[CLF] held out {len(val_eps)} episodes for validation: "
          f"{val_eps[:20]}{' ...' if len(val_eps) > 20 else ''}")

    train_ds = FrameClassifierDataset(
        ds, indices[train_mask], labels[train_mask],
        image_key=args.image_key, image_size=args.image_size, train=True,
    )
    val_ds = FrameClassifierDataset(
        ds, indices[val_mask], labels[val_mask],
        image_key=args.image_key, image_size=args.image_size, train=False,
    )

    # weight positive frames if very imbalanced (failure eps usually longer)
    pos = float(labels[train_mask].sum())
    neg = float((1 - labels[train_mask]).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=args.device)
    print(f"[CLF] train pos {int(pos)} / neg {int(neg)} -> "
          f"pos_weight = {pos_weight.item():.2f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    # ---------- model ----------
    model = ResNetClassifier(pretrained=not args.no_pretrained).to(args.device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    # override BCE with pos_weight version
    def loss_fn(logits, y):
        return F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight)
    # monkey-patch the bound used inside the train loop
    F_bce = F.binary_cross_entropy_with_logits

    def train_one_epoch_local(model, loader, opt, device):
        model.train()
        total = 0; loss_sum = 0.0; correct = 0
        for img, y in loader:
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(img)
            loss = F_bce(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = (logits.sigmoid() > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.numel()
                loss_sum += loss.item() * y.numel()
        return loss_sum / max(total, 1), correct / max(total, 1)

    # ---------- train ----------
    best_val_acc = 0.0
    history = []
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch_local(
            model, train_loader, opt, args.device)
        val_loss, val_acc, val_p, val_r = eval_loop(
            model, val_loader, args.device)
        history.append({
            "epoch": ep,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision": val_p, "val_recall": val_r,
        })
        print(f"[ep {ep:3d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"|  val: loss={val_loss:.4f} acc={val_acc:.3f} "
              f"P={val_p:.3f} R={val_r:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_key": args.image_key,
                "image_size": args.image_size,
                "imagenet_mean": list(FrameClassifierDataset.IMAGENET_MEAN),
                "imagenet_std": list(FrameClassifierDataset.IMAGENET_STD),
                "epoch": ep,
                "val_acc": val_acc,
                "args": vars(args),
            }, out / "best.pt")
            print(f"           -> saved best.pt (val_acc={val_acc:.3f})")

    # final dump
    torch.save({
        "model_state_dict": model.state_dict(),
        "image_key": args.image_key,
        "image_size": args.image_size,
        "imagenet_mean": list(FrameClassifierDataset.IMAGENET_MEAN),
        "imagenet_std": list(FrameClassifierDataset.IMAGENET_STD),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, out / "last.pt")
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out / "episode_labels.json", "w") as f:
        json.dump({str(k): v for k, v in per_ep.items()}, f, indent=2)
    print(f"[CLF] done. best val_acc = {best_val_acc:.3f}. "
          f"checkpoints in {out}")


if __name__ == "__main__":
    main()
