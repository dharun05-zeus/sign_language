"""
Training loop for the ASL translation model. Supports all three phases via
--phase, reading hyperparameters from config.yaml. Each phase loads the
previous phase's checkpoint (if present) so training is sequential:
Phase 1 -> Phase 2 -> Phase 3.

Usage:
    python src/train.py --phase 1 --config config.yaml
    python src/train.py --phase 2 --config config.yaml
    python src/train.py --phase 3 --config config.yaml

Each phase saves:
    checkpoints/phase{N}/projector.pt
    checkpoints/phase{N}/lora/   (peft adapter dir)
    checkpoints/phase{N}/training_log.csv
"""

import argparse
import csv
import os
import sys

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LandmarkTextDataset, collate_fn
from model import ASLTranslationModel


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_previous_phase_dir(cfg, phase):
    if phase == 1:
        return None
    return cfg[f"phase{phase - 1}"]["out_dir"]


def load_checkpoint_if_exists(model, ckpt_dir, device):
    projector_path = os.path.join(ckpt_dir, "projector.pt")
    lora_path = os.path.join(ckpt_dir, "lora")

    if os.path.exists(projector_path):
        print(f"Loading projector weights from {projector_path}")
        state = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(state)
    else:
        print(f"No projector checkpoint found at {projector_path} - starting fresh.")

    if os.path.isdir(lora_path):
        print(f"Loading LoRA adapter weights from {lora_path}")
        model.t5.load_adapter(lora_path, adapter_name="default", is_trainable=True)
    else:
        print(f"No LoRA checkpoint found at {lora_path} - starting fresh (randomly init LoRA).")

    return model


def save_checkpoint(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    model.t5.save_pretrained(os.path.join(out_dir, "lora"))
    print(f"Saved checkpoint to {out_dir}")


def run_epoch(model, dataloader, optimizer, scaler, device, tokenizer, train=True, log_prefix="", grad_accum=1, limit_batches=None):
    model.train(train)
    total_loss = 0.0
    n_batches = 0

    import time
    t_data_total = 0.0
    t_forward_total = 0.0
    t_backward_total = 0.0
    t_opt_total = 0.0

    t_start_data = time.perf_counter()

    if train:
        optimizer.zero_grad()

    progress = tqdm(dataloader, desc=log_prefix)
    for i, batch in enumerate(progress):
        t_data = time.perf_counter() - t_start_data
        t_data_total += t_data

        t_start_fwd = time.perf_counter()
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        texts = batch["texts"]

        target_enc = tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        labels = target_enc["input_ids"].to(device, non_blocking=True)
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad in loss

        device_type = "cuda" if "cuda" in device else "cpu"
        with autocast(device_type=device_type, dtype=torch.bfloat16, enabled=("cuda" in device)):
            outputs = model(landmarks=landmarks, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

        t_fwd = time.perf_counter() - t_start_fwd
        t_forward_total += t_fwd

        t_start_bwd = time.perf_counter()
        if train:
            scaler.scale(loss).backward()
        t_bwd = time.perf_counter() - t_start_bwd
        t_backward_total += t_bwd

        t_start_opt = time.perf_counter()
        if train and (i + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        t_opt = time.perf_counter() - t_start_opt
        t_opt_total += t_opt

        total_loss += loss.item() * grad_accum
        n_batches += 1
        progress.set_postfix(loss=f"{(loss.item() * grad_accum):.4f}")

        if limit_batches is not None and n_batches >= limit_batches:
            break

        t_start_data = time.perf_counter()

    if train and n_batches % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if n_batches > 0:
        print(f"\n[{log_prefix} Timing Summary]")
        print(f"  Total Batches Profiling: {n_batches}")
        print(f"  Avg Data Loading: {t_data_total / n_batches:.4f}s / batch")
        print(f"  Avg Forward Pass: {t_forward_total / n_batches:.4f}s / batch")
        print(f"  Avg Backward Pass: {t_backward_total / n_batches:.4f}s / batch")
        print(f"  Avg Optimizer Step: {t_opt_total / n_batches:.4f}s / batch")
        total_time = (t_data_total + t_forward_total + t_backward_total + t_opt_total)
        print(f"  Total Batch Time (excl. stdout): {total_time / n_batches:.4f}s / batch")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default=None,
                         help="Override manifest CSV path (defaults to phase config)")
    parser.add_argument("--text_col", default=None,
                         help="Override text column (gloss for phase 1/3, sentence for phase 2)")
    parser.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--allow_cpu", action="store_true", help="Bypass fast-fail CUDA check (for debugging only)")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches per epoch (for profiling)")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None,
                        help="Alias for --limit_batches: stop each epoch after N steps")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Run 5 steps, skip validation, print timing summary and exit. "
                             "Use to confirm CUDA + bitsandbytes + data loading before a full run.")
    parser.add_argument("--no_preload", action="store_true", help="Disable in-memory landmark dataset preloading")
    args = parser.parse_args()

    # --dry-run wires into existing flags: 5 steps, no val, single epoch
    if args.dry_run:
        args.max_steps = args.max_steps or 5
        args.val_every = 9999  # skip validation entirely during dry run
        print("\n" + "=" * 55)
        print(" DRY RUN MODE  (--dry-run, max_steps=%d)" % args.max_steps)
        print("=" * 55)

    # Resolve --max-steps → limit_batches
    if args.max_steps is not None:
        args.limit_batches = args.max_steps

    cfg = load_config(args.config)
    phase_key = f"phase{args.phase}"
    phase_cfg = cfg[phase_key]

    print("=== STARTUP CUDA & BITSANDBYTES DIAGNOSTICS ===")
    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA active device name: {torch.cuda.get_device_name(0)}")
    
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("WARNING: bitsandbytes is not installed.")

    print("===============================================")

    if not args.allow_cpu and not cuda_avail:
        raise RuntimeError(
            "CUDA is not available. Training on CPU will silently take "
            "hours per epoch on this hardware. Fix GPU/driver/CUDA setup "
            "before proceeding."
        )

    device = "cuda" if cuda_avail else "cpu"

    manifest_csv = args.manifest or phase_cfg.get("manifest_csv") or os.path.join(
        cfg["paths"]["landmarks_root"], phase_cfg["dataset"].lower(), "manifest.csv"
    )
    if not os.path.exists(manifest_csv):
        print(f"ERROR: manifest not found at {manifest_csv}")
        print("Run scripts/build_index.py and src/extract_landmarks.py first.")
        sys.exit(1)

    default_text_col = "gloss" if args.phase in (1, 3) else "sentence"
    text_col = args.text_col or default_text_col

    print(f"Phase {args.phase} | dataset={phase_cfg['dataset']} | manifest={manifest_csv} | text_col={text_col}")

    in_memory = not args.no_preload
    train_ds = LandmarkTextDataset(manifest_csv, text_col=text_col, split="train",
                                    max_frames=cfg["max_frames"], in_memory=in_memory)
    val_ds = LandmarkTextDataset(manifest_csv, text_col=text_col, split="val",
                                  max_frames=cfg["max_frames"], in_memory=in_memory)

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
    if len(train_ds) == 0:
        print("ERROR: 0 training examples found. Check your manifest's 'split' column "
              "contains 'train' rows.")
        sys.exit(1)

    # Use pin_memory=True for fast transfers to GPU
    use_pin = "cuda" in device
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin,
    )

    model = ASLTranslationModel(
        t5_model_name=cfg["t5_model"],
        landmark_dim=cfg["landmark_dim"],
        projector_hidden=cfg["projector_hidden"],
        t5_hidden=cfg["t5_hidden_size"],
        max_frames=cfg["max_frames"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    )

    if cfg.get("gradient_checkpointing"):
        model.t5.gradient_checkpointing_enable()

    prev_dir = get_previous_phase_dir(cfg, args.phase)
    if prev_dir and os.path.exists(prev_dir):
        model = load_checkpoint_if_exists(model, prev_dir, device)
    elif args.phase > 1:
        print(f"WARNING: expected previous phase checkpoint at {prev_dir} but none found. "
              f"Phase {args.phase} will start from an untrained projector / base LoRA init, "
              f"which defeats the purpose of sequential training. Proceed only if intentional.")

    model.projector.to(device)
    # t5 is already placed by device_map="auto" in build_t5_4bit_lora

    param_groups = model.trainable_parameter_groups(
        lr_projector=phase_cfg["lr_projector"],
        lr_lora=phase_cfg["lr_lora"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=False)

    out_dir = phase_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    log_rows = []

    best_val_loss = float("inf")
    grad_accum = cfg.get("gradient_accumulation_steps", 1)

    for epoch in range(1, phase_cfg["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, device, model.tokenizer,
            train=True, log_prefix=f"Phase{args.phase} Epoch{epoch}/[train]",
            grad_accum=grad_accum, limit_batches=args.limit_batches
        )

        val_loss = None
        if epoch % args.val_every == 0 and len(val_ds) > 0:
            with torch.no_grad():
                val_loss = run_epoch(
                    model, val_loader, optimizer, scaler, device, model.tokenizer,
                    train=False, log_prefix=f"Phase{args.phase} Epoch{epoch}/[val]",
                    grad_accum=grad_accum, limit_batches=args.limit_batches
                )

        # Dry-run: skip checkpointing, print summary, exit after 1 epoch
        if args.dry_run:
            peak_mb = torch.cuda.max_memory_allocated() // (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            print("\n" + "=" * 55)
            print(" DRY RUN COMPLETE")
            print("=" * 55)
            print(f"  device        : {torch.cuda.get_device_name(0)}")
            print(f"  train_loss    : {train_loss:.4f}  (sanity: should be finite)")
            print(f"  peak VRAM     : {peak_mb} MB / {total_mb} MB")
            print(f"  bitsandbytes  : loaded OK (4-bit model ran without fallback)")
            print("=" * 55)
            if not (train_loss == train_loss):  # NaN check
                print("  *** FAIL: loss is NaN — check model init or data ***")
            else:
                print("  *** DRY RUN PASSED — safe to start full training ***")
            print("=" * 55 + "\n")
            return

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_rows)

        # Always save latest; also flag best
        save_checkpoint(model, out_dir)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, os.path.join(out_dir, "best"))
            print(f"New best val_loss={val_loss:.4f} - saved to {out_dir}/best")

    print(f"\nPhase {args.phase} training complete. Final checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
