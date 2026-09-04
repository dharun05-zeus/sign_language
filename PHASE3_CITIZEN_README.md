# Phase 3 Instruction Guide: ASL Citizen Signer Diversity Fine-Tuning

> [!NOTE]
> **Why ASL Citizen? (Pivot from MS-ASL)**:
> Phase 3 originally targeted MS-ASL, but MS-ASL's 2019 YouTube-hosted videos suffer from ~90% link rot (privated, deleted, or bot-blocked videos). **ASL Citizen** is hosted directly by Microsoft as a single ZIP archive, eliminating link rot entirely.

---

## 1. Dataset Overview & Direct Download

- **Dataset**: ASL Citizen (~84,000 videos across 2,700 classes recorded by hundreds of crowdsourced signers in everyday environments).
- **Scope for Phase 3**: Top-100 most frequent sign classes (`ASL_Citizen_top100`) to keep landmark extraction and training duration fast and comparable in scope to WLASL100.
- **Direct Download Link (No YouTube Scraping)**:
  ```bash
  wget https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
  ```
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Directory Layout & Staging

Extract `ASL_Citizen.zip` and place the metadata splits and videos as follows:

```
studies/sign/
├── config_phase3_citizen.yaml     <- Phase 3 configuration & hyperparameters
├── PHASE3_CITIZEN_README.md       <- This guide
│
├── src/
│   ├── dataset_asl_citizen.py     <- Standalone ASL Citizen dataset loader & top-100 filter
│   ├── train_phase3_citizen.py    <- Main Phase 3 training script with BLEU guard
│   ├── dataset_how2sign.py        <- Used for How2Sign validation subset evaluation
│   ├── model.py                   <- ASL model architecture (from Phase 1)
│
├── checkpoints/
│   ├── phase2/                    <- PHASE 2 CHECKPOINTS (Loaded Read-Only)
│   │   ├── projector.pt
│   │   └── lora/
│   └── phase3_citizen/            <- PHASE 3 CITIZEN CHECKPOINTS (Saved here)
│
└── data/
    ├── asl_citizen/               <- Extracted ASL Citizen Dataset
    │   ├── splits/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── test.csv
    │   └── videos/                <- Raw .mp4 video clips
    ├── landmarks/
    │   └── asl_citizen/           <- Extracted 345-dim .npy landmark files
    └── how2sign/                  <- Held-out How2Sign validation subset (for BLEU guard)
        ├── val_manifest.csv
        └── val_transcripts.tsv
```

---

## 3. Execution Workflow

### Step 1: Extract 345-dim MediaPipe Landmarks
Extract 345-dim landmarks from the ASL Citizen videos (Pose 99 + Left Hand 63 + Right Hand 63 + Face 120):
```bash
python src/extract_landmarks.py --index data/asl_citizen/splits/train.csv --out_dir data/landmarks/asl_citizen --max_frames 150
python src/extract_landmarks.py --index data/asl_citizen/splits/val.csv --out_dir data/landmarks/asl_citizen --max_frames 150
```
*(Once extraction is finished, the raw `.mp4` video files can be deleted to save disk space).*

---

### Step 2: Run the Mandatory Dry-Run (8 Batches)
Run the dry-run command first. This executes exactly **8 batches** (2 complete gradient accumulation cycles), verifies GPU / bitsandbytes status, validates How2Sign baseline BLEU calibration, logs per-stage step timings, reports VRAM memory footprint, and exits without writing checkpoints:

```bash
python src/train_phase3_citizen.py --config config_phase3_citizen.yaml --dry-run
```

**Dry-Run Verification Checklist:**
- [x] **Device Confirmation**: Confirms GPU execution (`training on GPU: NVIDIA GeForce RTX 4050 Laptop GPU`).
- [x] **Quantization Integrity**: `bitsandbytes CUDA setup validated successfully`.
- [x] **Phase 2 Baseline Calibration**: Confirms How2Sign baseline BLEU-4 is computed and displayed.
- [x] **Step Timings**: Confirms data loading is near `0.00s` (due to RAM preloading) and forward/backward/optimizer step times are stable.
- [x] **VRAM Footprint**: Confirms memory usage is within the 6GB VRAM budget (~4.4GB expected).

---

### Step 3: Full Phase 3 Citizen Training Execution
Once the dry-run output is confirmed, launch the 10-epoch training loop:

```bash
python src/train_phase3_citizen.py --config config_phase3_citizen.yaml
```

Checkpoints will be saved to `checkpoints/phase3_citizen/` after each epoch, with the best validation checkpoint placed in `checkpoints/phase3_citizen/best/`.

---

## 4. Automated Catastrophic Forgetting Guard Mechanics

1. Before fine-tuning begins, `train_phase3_citizen.py` evaluates the loaded Phase 2 checkpoint on the How2Sign validation subset to establish a **Baseline BLEU-4**.
2. After each epoch of ASL Citizen fine-tuning (using 10x lower learning rates: Projector LR `5.0e-6`, LoRA LR `1.0e-5`), the script re-evaluates BLEU-4 on the How2Sign subset.
3. If `Baseline BLEU - Current BLEU > 2.0`:
   - A critical alert is logged: `🚨 CRITICAL ALERT: CATASTROPHIC FORGETTING TRIGGERED!`.
   - Training immediately halts.
   - The current epoch checkpoint is **discarded / not saved**.

---

## 5. Formal Sign-Off Criteria

Before deploying to live serving:
1. **ASL Citizen Top-1 Accuracy**: Target > 80% on the top-100 validation set.
2. **How2Sign Sentence BLEU Retention**: Final How2Sign BLEU-4 must remain within **2.0 points** of Phase 2.
3. **Multi-Signer Qualitative Spot-Check**: Manually review 15-20 predictions across diverse signers in the validation split.
