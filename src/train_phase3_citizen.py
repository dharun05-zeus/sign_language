"""
Phase 3 Training Script: ASL Citizen Signer Diversity Fine-Tuning.
Loads Phase 2 checkpoint as the starting point and fine-tunes on ASL Citizen
with 10x lower learning rate and an automatic catastrophic forgetting BLEU-4 guard.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_asl_citizen import ASLCitizenDataset, collate_fn
from dataset_how2sign import How2SignDataset
from model import ASLTranslationModel


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def verify_bitsandbytes_cuda():
    """Verify that bitsandbytes is correctly integrated with CUDA by running a mock quantized layer."""
    print("Verifying bitsandbytes CUDA setup...")
    try:
        import bitsandbytes as bnb
        dummy_layer = bnb.nn.Linear4bit(10, 10, bias=False).cuda()
        x = torch.randn(2, 10).cuda()
        with torch.no_grad():
            _ = dummy_layer(x)
        print(f"✅ bitsandbytes CUDA setup validated successfully (bnb version: {bnb.__version__}).")
        
        # Free memory occupied by dummy layer immediately
        del dummy_layer
        del x
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(
            f"bitsandbytes is not correctly utilizing CUDA. Quantized layers cannot be placed "
            f"on GPU. Fix GPU/driver/CUDA setup. Error detail: {e}"
        )


def load_phase2_read_only(model, ckpt_dir, device):
    """Load Phase 2 projector and LoRA weights in read-only mode."""
    best_dir = os.path.join(ckpt_dir, "best")
    load_dir = best_dir if os.path.exists(os.path.join(best_dir, "projector.pt")) else ckpt_dir
    
    projector_path = os.path.join(load_dir, "projector.pt")
    lora_path = os.path.join(load_dir, "lora")

    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Required Phase 2 projector checkpoint not found at: {projector_path}")
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"Required Phase 2 LoRA weights not found at: {lora_path}")

    print(f"Loading Phase 2 starting checkpoint from: {load_dir}")
    state = torch.load(projector_path, map_location=device)
    model.projector.load_state_dict(state)

    model.t5.load_adapter(lora_path, adapter_name="default", is_trainable=True)
    print("✅ Phase 2 checkpoint loaded successfully into projector and T5 LoRA.")
    return model


def save_checkpoint_phase3_citizen(model, out_dir):
    """Saves Phase 3 ASL Citizen checkpoint to target directory."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    model.t5.save_pretrained(os.path.join(out_dir, "lora"))
    print(f"Saved Phase 3 Citizen checkpoint to: {out_dir}")


def evaluate_how2sign_bleu(model, dataloader, device, num_beams=4):
    """Evaluates BLEU-4 on How2Sign validation subset to check sentence translation performance."""
    import sacrebleu
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
            landmarks = batch["landmarks"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            texts = batch["texts"]
            
            preds = model.generate(landmarks, mask, max_new_tokens=48, num_beams=num_beams)
            predictions.extend(preds)
            references.extend(texts)
            
    safe_preds = [p if p.strip() else "<empty>" for p in predictions]
    safe_refs = [r if r.strip() else "<empty>" for r in references]
    
    bleu_result = sacrebleu.corpus_bleu(safe_preds, [safe_refs])
    return bleu_result.score, safe_preds, safe_refs


def run_epoch(model, dataloader, optimizer, scaler, device, tokenizer, train=True, 
              log_prefix="", grad_accum=1, limit_batches=None, max_grad_norm=1.0):
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
        labels[labels == tokenizer.pad_token_id] = -100

        device_type = "cuda" if "cuda" in device else "cpu"
        with autocast(device_type=device_type, dtype=torch.float16, enabled=("cuda" in device)):
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
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

    # Step any remaining gradients at epoch end
    if train and n_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if n_batches > 0:
        print(f"\n[{log_prefix} Timing Summary]")
        print(f"  Total Batches: {n_batches}")
        print(f"  Avg Data Loading:   {t_data_total / n_batches:.4f}s / batch")
        print(f"  Avg Forward Pass:   {t_forward_total / n_batches:.4f}s / batch")
        print(f"  Avg Backward Pass:  {t_backward_total / n_batches:.4f}s / batch")
        print(f"  Avg Optimizer Step: {t_opt_total / n_batches:.4f}s / batch")
        total_time = (t_data_total + t_forward_total + t_backward_total + t_opt_total)
        print(f"  Total Step Time:    {total_time / n_batches:.4f}s / batch")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="ASL Signer Diversity Training (Phase 3 - ASL Citizen)")
    parser.add_argument("--config", default="config_phase3_citizen.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run 8 batches only, report timing & VRAM, exit")
    parser.add_argument("--no-preload", action="store_true", help="Disable in-memory dataset preloading")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. HARD CUDA ASSERTION
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU is strictly required for Phase 3 training. "
            "No CPU fallback permitted for 4-bit quantized model pipeline."
        )

    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    print(f"Hard device check passed: training on GPU: {device_name}")

    # 2. BITSANDBYTES INTEGRATION VERIFICATION
    verify_bitsandbytes_cuda()

    # 3. INITIALIZE MODEL ARCHITECTURE
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

    # 4. LOAD PHASE 2 CHECKPOINT (READ-ONLY)
    phase2_dir = cfg["phase2_checkpoint_dir"]
    model = load_phase2_read_only(model, phase2_dir, device)
    model.projector.to(device)

    # 5. PREPARE HOW2SIGN HELD-OUT SUBSET & CALIBRATE BASELINE BLEU-4
    how2sign_cfg = cfg["how2sign_eval"]
    how2sign_loader = None
    baseline_bleu = None

    if os.path.exists(how2sign_cfg["val_manifest"]) and os.path.exists(how2sign_cfg["val_transcripts"]):
        print("\nLoading How2Sign validation subset for catastrophic forgetting guard...")
        h2s_val_ds = How2SignDataset(
            manifest_csv=how2sign_cfg["val_manifest"],
            transcripts_tsv=how2sign_cfg["val_transcripts"],
            split="val",
            max_frames=cfg["max_frames"],
            in_memory=True,
        )
        subset_limit = how2sign_cfg.get("subset_limit", 100)
        if len(h2s_val_ds) > subset_limit:
            indices = list(range(subset_limit))
            h2s_val_subset = Subset(h2s_val_ds, indices)
        else:
            h2s_val_subset = h2s_val_ds
            
        how2sign_loader = DataLoader(
            h2s_val_subset, batch_size=cfg["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=True
        )

        print("Calibrating baseline How2Sign sentence BLEU-4 before Phase 3 fine-tuning...")
        baseline_bleu, sample_preds, sample_refs = evaluate_how2sign_bleu(model, how2sign_loader, device)
        print(f"✅ Calibrated Baseline How2Sign BLEU-4: {baseline_bleu:.2f}")
        print("Sample Phase 2 baseline predictions:")
        for sp, sr in zip(sample_preds[:3], sample_refs[:3]):
            print(f"  Target: {sr}")
            print(f"  Pred:   {sp}\n")
    else:
        print("⚠️ Warning: How2Sign validation data not found at configured paths. "
              "Catastrophic forgetting guard will log warnings but cannot compute BLEU.")

    # 6. LOAD ASL CITIZEN DATASET
    citizen_cfg = cfg["asl_citizen"]
    in_memory = not args.no_preload

    print("\nLoading ASL Citizen training split...")
    train_ds = ASLCitizenDataset(
        metadata_csv=citizen_cfg.get("train_csv"),
        landmarks_dir=citizen_cfg["landmarks_dir"],
        split="train",
        top_n_classes=citizen_cfg.get("top_n_classes", 100),
        max_frames=cfg["max_frames"],
        in_memory=in_memory,
    )

    print("\nLoading ASL Citizen validation split...")
    val_ds = ASLCitizenDataset(
        metadata_csv=citizen_cfg.get("val_csv"),
        landmarks_dir=citizen_cfg["landmarks_dir"],
        split="val",
        top_n_classes=citizen_cfg.get("top_n_classes", 100),
        max_frames=cfg["max_frames"],
        in_memory=in_memory,
    )

    print(f"\nASL Citizen dataset splits: Train={len(train_ds)} | Val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # 7. OPTIMIZER & 10x REDUCED LEARNING RATES
    param_groups = model.trainable_parameter_groups(
        lr_projector=citizen_cfg["lr_projector"],
        lr_lora=citizen_cfg["lr_lora"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=cfg.get("fp16", True))

    out_dir = cfg["phase3_out_dir"]
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    bleu_drop_threshold = cfg.get("bleu_drop_threshold", 2.0)

    # 8. MANDATORY DRY-RUN EXECUTION
    if args.dry_run:
        print("\n================ STARTING DRY RUN (8 BATCHES) ================")
        print(f"Device: {device_name}")
        print(f"Projector LR: {citizen_cfg['lr_projector']} | LoRA LR: {citizen_cfg['lr_lora']}")
        
        run_epoch(
            model, train_loader, optimizer, scaler, device, model.tokenizer,
            train=True, log_prefix="DryRun/[train]",
            grad_accum=grad_accum, limit_batches=8, max_grad_norm=max_grad_norm
        )
        
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"\nVRAM Memory Utilization Profile:")
        print(f"  Current Allocated: {allocated_mem:.2f} MB")
        print(f"  Max Allocated:     {max_allocated_mem:.2f} MB")
        print(f"  Reserved Cache:    {reserved_mem:.2f} MB")
        print("==============================================================")
        print("Dry run completed successfully. Checkpoints were NOT modified.")
        sys.exit(0)

    # 9. FULL PHASE 3 TRAINING LOOP WITH CATASTROPHIC FORGETTING GUARD
    print(f"\nStarting Phase 3 ASL Citizen training loop for {citizen_cfg['epochs']} epochs.")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log_phase3_citizen.csv")
    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, citizen_cfg["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, device, model.tokenizer,
            train=True, log_prefix=f"Phase3_Citizen Epoch {epoch}/[train]",
            grad_accum=grad_accum, max_grad_norm=max_grad_norm
        )

        val_loss = None
        if epoch % citizen_cfg["val_every"] == 0 and len(val_ds) > 0:
            with torch.no_grad():
                val_loss = run_epoch(
                    model, val_loader, optimizer, scaler, device, model.tokenizer,
                    train=False, log_prefix=f"Phase3_Citizen Epoch {epoch}/[val]",
                    grad_accum=grad_accum, max_grad_norm=max_grad_norm
                )

        print(f"\nEpoch {epoch} Results: train_loss={train_loss:.4f}"
              + (f" | val_loss={val_loss:.4f}" if val_loss is not None else ""))

        # CATASTROPHIC FORGETTING CHECK
        if how2sign_loader is not None and baseline_bleu is not None:
            current_bleu, _, _ = evaluate_how2sign_bleu(model, how2sign_loader, device)
            bleu_drop = baseline_bleu - current_bleu
            print(f"Catastrophic Forgetting Check (Epoch {epoch}):")
            print(f"  Baseline How2Sign BLEU-4 : {baseline_bleu:.2f}")
            print(f"  Current How2Sign BLEU-4  : {current_bleu:.2f}")
            print(f"  BLEU-4 Drop Magnitude    : {bleu_drop:.2f} (Limit: {bleu_drop_threshold:.2f})")

            if bleu_drop > bleu_drop_threshold:
                print("\n" + "!" * 80)
                print(f"🚨 CRITICAL ALERT: CATASTROPHIC FORGETTING TRIGGERED!")
                print(f"How2Sign BLEU-4 dropped by {bleu_drop:.2f} points (exceeded threshold {bleu_drop_threshold:.2f}).")
                print("Halting training immediately. Checkpoint for this epoch will NOT be saved.")
                print("!" * 80 + "\n")
                break
            else:
                print("✅ How2Sign sentence translation ability preserved within safety margins.")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "how2sign_bleu": current_bleu if (how2sign_loader and baseline_bleu) else "N/A"
        })
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "how2sign_bleu"])
            writer.writeheader()
            writer.writerows(log_rows)

        # Save checkpoint
        save_checkpoint_phase3_citizen(model, out_dir)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint_phase3_citizen(model, os.path.join(out_dir, "best"))
            print(f"New best Phase 3 Citizen checkpoint saved to: {out_dir}/best")

    print(f"\nPhase 3 ASL Citizen training execution finished. Final checkpoints stored at: {out_dir}")


if __name__ == "__main__":
    main()
