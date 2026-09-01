# Phase 3 Instruction Guide: MS-ASL Signer Diversity Fine-Tuning

This guide is for the teammate setting up and executing **Phase 3** training on the target CUDA-capable machine (RTX 4050 Laptop GPU, 6GB VRAM).

> [!IMPORTANT]
> **MS-ASL Dataset License Notice**:
> The MS-ASL dataset metadata and videos are released under a **Microsoft Research Use Only** license. Ensure this license is observed for any public redistribution or commercial evaluations.

---

## 1. Overview & Objectives
- **Goal**: Generalize the model across **200+ real-world signers** (MS-ASL) so the translation pipeline is robust across different signing styles, body sizes, and lighting conditions.
- **Critical Risk**: Catastrophic forgetting of Phase 2's sentence-level translation capabilities.
- **Mitigations**:
  1. **10x Lower Learning Rates**: Projector LR = `5.0e-6`, LoRA LR = `1.0e-5` (10x lower than Phase 2).
  2. **Automated BLEU-4 Guard**: Automatically measures sentence BLEU on a held-out How2Sign validation subset after every epoch. If BLEU drops > `2.0` points from baseline, training automatically halts immediately without saving that epoch's checkpoint.

---

## 2. Directory Structure & Staging

Organize the repository so Phase 2 checkpoints are available in read-only mode, and the MS-ASL100 data and How2Sign validation subset are in place:

```
studies/sign/
├── config_phase3.yaml         <- Phase 3 configuration & hyperparameters
├── src/
│   ├── dataset_msasl.py       <- Standalone MS-ASL Dataset loader & synonym parser
│   ├── train_phase3.py        <- Main Phase 3 training script with BLEU guard
│   ├── dataset_how2sign.py    <- Used for How2Sign validation subset evaluation
│   ├── model.py               <- ASL model architecture (from Phase 1)
├── checkpoints/
│   ├── phase2/                <- PHASE 2 CHECKPOINTS (Loaded Read-Only)
│   │   ├── projector.pt
│   │   └── lora/
│   └── phase3/                <- PHASE 3 CHECKPOINTS (Saved here)
├── data/
│   ├── msasl/                 <- MS-ASL Metadata JSONs
│   │   ├── MSASL_train.json
│   │   ├── MSASL_val.json
│   │   ├── MSASL_test.json
│   │   ├── MSASL_classes.json
│   │   └── MSASL_synonym.json
│   ├── landmarks/
│   │   └── msasl100/          <- 345-dim extracted landmark .npy files
│   └── how2sign/              <- How2Sign validation subset (for BLEU guard)
│       ├── val_manifest.csv
│       └── val_transcripts.tsv
```

---

## 3. Training Execution Workflow

Ensure you are inside the `asl-env` environment (Python `3.10.11`):
```bash
conda activate asl-env
```

### Step A: Mandatory Dry-Run Execution
Run the mandatory `--dry-run` flag first. This runs exactly **8 batches** (2 complete gradient accumulation cycles), verifies device and bitsandbytes CUDA status, tests the in-memory preloader, logs per-stage step timings, reports VRAM memory footprint, and exits without writing any checkpoints:

```bash
python src/train_phase3.py --config config_phase3.yaml --dry-run
```

**Dry-Run Verification Checklist:**
1. **Device Confirmation**: Confirms GPU execution (`training on GPU: NVIDIA GeForce RTX 4050 Laptop GPU`).
2. **Quantization Integrity**: Confirms `bitsandbytes` CUDA sanity test passes and dummy layer VRAM is freed.
3. **Phase 2 Baseline Calibration**: Confirms How2Sign baseline BLEU-4 is computed and displayed.
4. **Step Timings**: Confirms data loading is near `0.00s` (due to RAM preloading) and forward/backward/optimizer step times are stable.
5. **VRAM Footprint**: Confirms memory usage is within the 6GB VRAM budget (~4.4GB expected).

### Step B: Full Phase 3 Training Execution
Once the dry-run output is confirmed, launch the 10-epoch training loop:

```bash
python src/train_phase3.py --config config_phase3.yaml
```

Checkpoints will be saved to `checkpoints/phase3/` after each epoch, with the best validation checkpoint placed in `checkpoints/phase3/best/`.

---

## 4. Automated Catastrophic Forgetting Guard Mechanics

1. Before training begins, `train_phase3.py` evaluates the loaded Phase 2 checkpoint on the How2Sign validation subset to establish a **Baseline BLEU-4**.
2. After each epoch of MS-ASL100 fine-tuning, the script re-evaluates BLEU-4 on the How2Sign subset.
3. If `Baseline BLEU - Current BLEU > 2.0`:
   - A critical alert is logged: `🚨 CRITICAL ALERT: CATASTROPHIC FORGETTING TRIGGERED!`.
   - The script immediately stops training.
   - The current epoch checkpoint is **discarded / not saved**.

---

## 5. Formal Sign-Off Criteria (Human Verification)

Passing metrics alone are not sufficient sign-off. Before expanding to MS-ASL300 or deploying:
1. **MS-ASL100 Top-1 Accuracy**: Must achieve **> 80%** accuracy on the held-out MS-ASL100 validation set.
2. **How2Sign BLEU-4 Retention**: Final How2Sign sentence BLEU-4 must be within **2.0 points** of Phase 2's benchmark.
3. **Multi-Signer Qualitative Spot-Check**: Manually review 15-20 predictions across diverse signers in the validation split (different lighting, backgrounds, skin tones, and signing speeds) to confirm generalizability.
