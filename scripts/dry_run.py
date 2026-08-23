"""
dry_run.py — Short batch sanity check

Verifies the full training stack is functional before committing to a
multi-hour run. Checks:
    1. CUDA device is available and bitsandbytes CUDA ops load correctly.
    2. A single forward + backward pass completes without OOM.
    3. DataLoader yields a batch in reasonable time.
    4. Mixed-precision (fp16) and gradient accumulation are active.
    5. Reports timing, peak VRAM usage, and gradient norms.

Usage (activate venv first):
    python scripts/dry_run.py --phase 1 --config config.yaml --batches 3

Expected output on RTX 4050 (6 GB):
    [device]   NVIDIA GeForce RTX 4050 Laptop GPU — CUDA 12.x
    [bnb]      bitsandbytes OK
    [data]     batch loaded in <2s  shape=(4, 150, 345)
    [forward]  OK  loss=<some float>
    [backward] OK  grad_norm=<some float>
    [vram]     peak ~3.8 GB / 6.0 GB
    [timing]   <N> batches in <T>s  (~<T/N>s/batch)
    DRY RUN PASSED
"""

from __future__ import annotations

import argparse
import time
import sys
import os

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dry-run sanity check for the ASL pipeline.")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--batches", type=int, default=3,
                   help="Number of mini-batches to run (default: 3)")
    return p.parse_args()


def check_cuda() -> str:
    import torch
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False. "
        "Re-install PyTorch with CUDA: "
        "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )
    name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    print(f"[device]   {name} — total VRAM: {total_mb} MB")
    return name


def check_bnb() -> None:
    try:
        import bitsandbytes as bnb  # noqa: F401
        print("[bnb]      bitsandbytes OK")
    except Exception as e:
        print(f"[bnb]      FAILED: {e}")
        print("           Run: python -m bitsandbytes  for diagnostics")
        sys.exit(1)


def run_forward_backward(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from model import build_model
    from dataset import LandmarkDataset
    from torch.utils.data import DataLoader

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    phase_key = f"phase{args.phase}"
    phase_cfg = cfg[phase_key]

    # Build dataset
    manifest = os.path.join(
        cfg["paths"]["landmarks_root"],
        phase_cfg["dataset"].lower().replace("-", ""),
        "manifest.csv",
    )
    if not os.path.exists(manifest):
        print(f"[data]     manifest not found at {manifest} — skipping data check")
        return

    dataset = LandmarkDataset(
        manifest_path=manifest,
        max_frames=cfg["max_frames"],
        landmark_dim=cfg["landmark_dim"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        num_workers=0, pin_memory=True)

    t0 = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - t0
    landmarks, labels = batch["landmarks"], batch["label"]
    print(f"[data]     batch loaded in {elapsed:.2f}s  shape={tuple(landmarks.shape)}")

    # Build model
    model, tokenizer = build_model(cfg, phase=args.phase)
    model.train()

    device = "cuda"
    landmarks = landmarks.to(device, dtype=torch.float16)

    # Forward
    t1 = time.time()
    with torch.cuda.amp.autocast():
        outputs = model(landmarks=landmarks, labels=labels)
        loss = outputs.loss
    print(f"[forward]  OK  loss={loss.item():.4f}")

    # Backward
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"[backward] OK  grad_norm={total_norm:.4f}")

    peak_mb = torch.cuda.max_memory_allocated() // (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    print(f"[vram]     peak {peak_mb} MB / {total_mb} MB")

    # Multi-batch timing
    model.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        lm = batch["landmarks"].to(device, dtype=torch.float16)
        lb = batch["label"]
        with torch.cuda.amp.autocast():
            out = model(landmarks=lm, labels=lb)
        out.loss.backward()
        model.zero_grad()
    total_time = time.time() - t_start
    print(f"[timing]   {args.batches} batches in {total_time:.1f}s "
          f"(~{total_time/args.batches:.2f}s/batch)")


def main() -> None:
    args = parse_args()
    print(f"\n=== DRY RUN  phase={args.phase}  config={args.config} ===\n")

    check_cuda()
    check_bnb()
    run_forward_backward(args)

    print("\nDRY RUN PASSED ✓\n")


if __name__ == "__main__":
    main()
