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


def run_epoch(model, dataloader, optimizer, scaler, device, tokenizer, train=True, log_prefix=""):
    model.train(train)
    total_loss = 0.0
    n_batches = 0

    progress = tqdm(dataloader, desc=log_prefix)
    for batch in progress:
        landmarks = batch["landmarks"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]

        target_enc = tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        labels = target_enc["input_ids"].to(device)
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad in loss

        if train:
            optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(landmarks=landmarks, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

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
    args = parser.parse_args()

    cfg = load_config(args.config)
    phase_key = f"phase{args.phase}"
    phase_cfg = cfg[phase_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available - training will be extremely slow / may OOM on CPU.")

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

    train_ds = LandmarkTextDataset(manifest_csv, text_col=text_col, split="train",
                                    max_frames=cfg["max_frames"])
    val_ds = LandmarkTextDataset(manifest_csv, text_col=text_col, split="val",
                                  max_frames=cfg["max_frames"])

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
    if len(train_ds) == 0:
        print("ERROR: 0 training examples found. Check your manifest's 'split' column "
              "contains 'train' rows.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
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
    scaler = GradScaler(enabled=cfg.get("fp16", True))

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
        )

        val_loss = None
        if epoch % args.val_every == 0 and len(val_ds) > 0:
            with torch.no_grad():
                val_loss = run_epoch(
                    model, val_loader, optimizer, scaler, device, model.tokenizer,
                    train=False, log_prefix=f"Phase{args.phase} Epoch{epoch}/[val]",
                )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}"
              + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

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
