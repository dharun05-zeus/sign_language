"""
Phase 2 Training Script: ASL-to-English Sentence Translation on How2Sign.
Loads Phase 1 checkpoint as a read-only starting point and fine-tunes on sentence data.
"""

import argparse
import csv
import gc
import os
import sys
import time
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_how2sign import How2SignDataset, collate_fn
from model import ASLTranslationModel


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def verify_bitsandbytes_cuda():
    """Verify that bitsandbytes is correctly integrated with CUDA by running a mock quantized layer."""
    print("Verifying bitsandbytes CUDA setup...")
    try:
        import bitsandbytes as bnb
        # Instantiate a dummy 4-bit layer and run on GPU
        dummy_layer = bnb.nn.Linear4bit(10, 10, bias=False).cuda()
        x = torch.randn(2, 10).cuda()
        with torch.no_grad():
            _ = dummy_layer(x)
        print(f"[OK] bitsandbytes CUDA setup validated successfully (bnb version: {bnb.__version__}).")
        
        # Explicitly free memory occupied by the dummy layer
        del dummy_layer
        del x
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(
            f"bitsandbytes is not correctly utilizing CUDA. Quantized layers cannot be placed "
            f"on GPU. Fix GPU/driver/CUDA setup. Error detail: {e}"
        )


def run_spot_check(model, wlasl_manifest, device, cfg):
    """Runs a read-only check of the Phase 1 checkpoint on WLASL100 validation samples."""
    print("\n=== PHASE 1 INTEGRITY SPOT-CHECK ===")
    if not os.path.exists(wlasl_manifest):
        print(f"Skipping WLASL100 spot-check: manifest not found at {wlasl_manifest}.")
        print("Proceeding to Phase 2. This is normal if this machine only hosts the How2Sign dataset.")
        print("=====================================\n")
        return

    # Attempt to import original Phase 1 dataset loader
    try:
        from dataset import LandmarkTextDataset
    except ImportError:
        print("Skipping WLASL100 spot-check: could not import dataset.py. Proceeding.")
        print("=====================================\n")
        return

    try:
        print(f"Loading spot-check samples from WLASL100 manifest: {wlasl_manifest}")
        # Initialize in-memory dataset with just a subset (or load normally without caching for speed)
        val_ds = LandmarkTextDataset(wlasl_manifest, text_col="gloss", split="val",
                                      max_frames=cfg["max_frames"], in_memory=False)
        if len(val_ds) == 0:
            print("WLASL100 validation split is empty. Skipping spot-check.")
            print("=====================================\n")
            return

        model.eval()
        samples_to_check = min(5, len(val_ds))
        print(f"Running inference on {samples_to_check} random validation glosses:")
        print("-" * 50)
        
        # Load a few random items
        indices = torch.randperm(len(val_ds))[:samples_to_check].tolist()
        
        with torch.no_grad():
            for idx in indices:
                item = val_ds[idx]
                landmarks = item["landmarks"].unsqueeze(0).to(device)
                mask = item["attention_mask"].unsqueeze(0).to(device)
                target = item["text"]
                
                preds = model.generate(landmarks, mask, max_new_tokens=16, num_beams=4)
                pred_text = preds[0] if preds else "<empty>"
                print(f"  Target Gloss:    {target:<15} | Predicted: {pred_text}")
                
        print("\n[OK] Phase 1 spot-check completed successfully. Network loaded correctly.")
        print("=====================================\n")
    except Exception as e:
        print(f"[WARNING] Warning during WLASL100 spot-check: {e}")
        print("Review setup before starting full training. Proceeding to training script setup.")
        print("=====================================\n")


def run_epoch(model, dataloader, optimizer, scaler, device, tokenizer, train=True, 
              log_prefix="", grad_accum=1, limit_batches=None, dtype=torch.float16):
    model.train(train)
    total_loss = 0.0
    n_batches = 0

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
        with autocast(device_type=device_type, dtype=dtype, enabled=("cuda" in device)):
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
            # Gradient clipping at max_norm=1.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

    # Apply any remaining gradients at the end of the epoch
    if train and n_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def load_phase1_read_only(model, ckpt_dir, device):
    """Load Phase 1 projector and LoRA weights in read-only mode."""
    projector_path = os.path.join(ckpt_dir, "projector.pt")
    lora_path = os.path.join(ckpt_dir, "lora")

    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Required Phase 1 projector checkpoint not found at: {projector_path}")
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"Required Phase 1 LoRA weights not found at: {lora_path}")

    print(f"Loading Phase 1 projector weights from: {projector_path}")
    state = torch.load(projector_path, map_location=device)
    model.projector.load_state_dict(state)

    print(f"Loading Phase 1 LoRA adapter weights from: {lora_path}")
    model.t5.load_adapter(lora_path, adapter_name="default", is_trainable=True)
    
    return model


def save_checkpoint_phase2(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    model.t5.save_pretrained(os.path.join(out_dir, "lora"))
    print(f"Saved Phase 2 checkpoint to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="ASL Sentence Translation Fine-Tuning (Phase 2)")
    parser.add_argument("--config", default="config_phase2.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run 8 batches only, skip saving checkpoints, print profile")
    parser.add_argument("--no-preload", action="store_true", help="Disable in-memory landmark dataset preloading")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # 1. HARD CUDA & DRIVER ASSERTION
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU is required for Phase 2 training. "
            "CPU execution is not supported for 4-bit quantized model pipelines."
        )

    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    print(f"Hard device check passed: training on GPU: {device_name}")

    # 2. BITSANDBYTES CUDA INTEGRATION CHECK
    verify_bitsandbytes_cuda()

    # Initialize model architecture
    print("\nInitializing model architecture...")
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

    # 3. LOAD PHASE 1 CHECKPOINT IN READ-ONLY MODE
    phase1_dir = cfg["phase1_checkpoint_dir"]
    model = load_phase1_read_only(model, phase1_dir, device)

    # Move projector to GPU (t5 layers are already mapped by device_map="auto")
    model.projector.to(device)

    # 4. RUN RUNTIME SPOT-CHECK
    run_spot_check(model, cfg.get("wlasl_val_manifest", ""), device, cfg)

    # Load How2Sign Datasets
    how2sign_cfg = cfg["how2sign"]
    in_memory = not args.no_preload

    print("\nLoading How2Sign training data...")
    train_ds = How2SignDataset(
        manifest_csv=how2sign_cfg["train_manifest"],
        transcripts_tsv=how2sign_cfg["train_transcripts"],
        split="train",
        max_frames=cfg["max_frames"],
        in_memory=in_memory
    )
    
    print("\nLoading How2Sign validation data...")
    val_ds = How2SignDataset(
        manifest_csv=how2sign_cfg["val_manifest"],
        transcripts_tsv=how2sign_cfg["val_transcripts"],
        split="val",
        max_frames=cfg["max_frames"],
        in_memory=in_memory
    )

    print(f"\nHow2Sign dataset splits: Train={len(train_ds)} | Val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Parameter groups and Optimizer
    param_groups = model.trainable_parameter_groups(
        lr_projector=how2sign_cfg["lr_projector"],
        lr_lora=how2sign_cfg["lr_lora"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    # Determine precision / dtype dynamically
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("Using bfloat16 mixed precision training (more stable for T5).")
        dtype = torch.bfloat16
        scaler = GradScaler(enabled=False)
    elif cfg.get("fp16", True):
        print("Using float16 mixed precision training.")
        dtype = torch.float16
        scaler = GradScaler(enabled=True)
    else:
        print("Using float32 training.")
        dtype = torch.float32
        scaler = GradScaler(enabled=False)

    out_dir = cfg["phase2_out_dir"]
    grad_accum = cfg.get("gradient_accumulation_steps", 1)

    if args.dry_run:
        print("\n================ STARTING DRY RUN (8 BATCHES) ================")
        print(f"Device Name: {device_name}")
        
        # Dry-run execution
        run_epoch(
            model, train_loader, optimizer, scaler, device, model.tokenizer,
            train=True, log_prefix="DryRun/[train]",
            grad_accum=grad_accum, limit_batches=8, dtype=dtype
        )
        
        # Memory profile reporting
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"\nVRAM Memory Utilization Profile:")
        print(f"  Current Allocated: {allocated_mem:.2f} MB")
        print(f"  Max Allocated:     {max_allocated_mem:.2f} MB")
        print(f"  Reserved Cache:    {reserved_mem:.2f} MB")
        print("==============================================================")
        print("Dry run completed successfully. Review timings/VRAM before starting full epoch training.")
        sys.exit(0)

    # Full training loop
    print(f"\nStarting Phase 2 full training loop for {how2sign_cfg['epochs']} epochs.")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log_phase2.csv")
    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, how2sign_cfg["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, device, model.tokenizer,
            train=True, log_prefix=f"Phase2 Epoch {epoch}/[train]",
            grad_accum=grad_accum, dtype=dtype
        )

        val_loss = None
        if epoch % how2sign_cfg["val_every"] == 0 and len(val_ds) > 0:
            with torch.no_grad():
                val_loss = run_epoch(
                    model, val_loader, optimizer, scaler, device, model.tokenizer,
                    train=False, log_prefix=f"Phase2 Epoch {epoch}/[val]",
                    grad_accum=grad_accum, dtype=dtype
                )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}"
              + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_rows)

        # Save latest epoch checkpoint; also track and save best
        save_checkpoint_phase2(model, out_dir)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint_phase2(model, os.path.join(out_dir, "best"))
            print(f"New best Phase 2 val_loss={val_loss:.4f} - saved to {out_dir}/best")

    print(f"\nPhase 2 training complete. Final checkpoint saved to: {out_dir}")


if __name__ == "__main__":
    main()
