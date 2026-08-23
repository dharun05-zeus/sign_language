# Phase 2 Instruction Guide: How2Sign Sentence Translation Fine-Tuning

This guide is for the teammate setting up and executing Phase 2 training for the ASL sentence translation pipeline on the target CUDA-capable device (RTX 4050 GPU, 6GB VRAM).

---

## 1. Directory Structure Setup

Before running training, organize the project directory so that Phase 1 checkpoints are loaded in read-only mode, and the How2Sign data is mapped correctly:

```
studies/sign/
├── config_phase2.yaml        <- Hyperparameters and path configurations
├── src/
│   ├── dataset_how2sign.py    <- Standalone How2Sign Dataset class
│   ├── train_phase2.py        <- Main Phase 2 training script
│   ├── model.py               <- ASL model architecture (from Phase 1)
│   ├── dataset.py             <- WLASL dataset loader (for read-only spot check)
├── checkpoints/
│   ├── phase1/                <- PHASE 1 CHECKPOINTS (Loaded Read-Only)
│   │   ├── projector.pt
│   │   └── lora/
│   └── phase2/                <- PHASE 2 CHECKPOINTS (Will be written here)
├── data/
│   ├── how2sign/              <- How2Sign Dataset Folder
│   │   ├── train_manifest.csv <- Manifest mapping clip IDs to .npy paths
│   │   ├── train_transcripts.tsv <- English sentence translations
│   │   ├── val_manifest.csv
│   │   └── val_transcripts.tsv
│   └── landmarks/
│       └── wlasl100/          <- WLASL100 Validation Data (for spot checking, Optional)
│           └── manifest.csv
```

---

## 2. Generating Manifests and Transcripts (Phase 2 Indexing)

If your dataset does not already have pre-built `train_manifest.csv` or `train_transcripts.tsv` files matching the project format, you can automatically generate them from your folder of `.npy` landmarks and official How2Sign TSVs. 

Run the indexing script for the **train** and **validation** sets:

```bash
# 1. Build the train index
python scripts/build_how2sign_index.py \
    --npy_dir data/landmarks/how2sign/train \
    --tsv_path data/how2sign/how2sign_realign_train.tsv \
    --split train \
    --out_manifest data/how2sign/train_manifest.csv \
    --out_transcripts data/how2sign/train_transcripts.tsv

# 2. Build the validation index
python scripts/build_how2sign_index.py \
    --npy_dir data/landmarks/how2sign/val \
    --tsv_path data/how2sign/how2sign_realign_val.tsv \
    --split val \
    --out_manifest data/how2sign/val_manifest.csv \
    --out_transcripts data/how2sign/val_transcripts.tsv
```

This maps files on disk, verifies they exist, and outputs the manifest files in the layout required by the training script.

---

## 3. Dependencies and virtual environment
Ensure you are inside the `asl-env` virtual environment (running Python `3.10.11`) and all dependencies are installed:
```bash
conda activate asl-env
pip install -r requirements.txt
```

---

## 3. Training Execution Sequence

### Step A: Mandatory Diagnostics & Dry Run
To ensure your GPU is correctly recognized, `bitsandbytes` can successfully allocate quantized memory, and there are no path mismatches, run the **mandatory dry-run** command. This runs exactly **8 batches** (spanning 2 full gradient accumulation cycles), logs execution timings, profiles GPU memory, and terminates without writing checkpoints:

```bash
python src/train_phase2.py --config config_phase2.yaml --dry-run
```

**What to verify in the console output:**
1. **Device Confirmation**: Confirms training on GPU (`Hard device check passed: training on GPU: NVIDIA GeForce RTX 4050 Laptop GPU`).
2. **bitsandbytes Integration**: Verifies quantization (`bitsandbytes CUDA setup validated successfully`).
3. **Phase 1 Integrity Check**: Confirms the read-only checkpoint load succeeds and outputs a few translation evaluations on WLASL100 val samples.
4. **VRAM Memory Utilization**: Shows current and max allocated memory. Ensure it sits safely within the 6GB VRAM budget (~4.4GB expected).
5. **Execution Timings**: Confirms data loading is near `0.00s` (due to in-memory preloading) and verifies the forward/backward/optimizer step times.

### Step B: Full Training Execution
Once the dry-run logs are validated, initiate the full 20-epoch sentence-translation training:

```bash
python src/train_phase2.py --config config_phase2.yaml
```
Checkpoints will be written to `checkpoints/phase2/` after each epoch, and the best validation checkpoint will be stored in `checkpoints/phase2/best/`.

---

## 4. Post-Training Evaluation

After Phase 2 training is completed, run the evaluations and spot-checks to sign off on model performance.

### Quantitative Metrics Evaluation
Run the evaluation script on the How2Sign validation split. The target thresholds to achieve are:
- **BLEU-4** > 10 (SOTA is 12.39)
- **ROUGE-L** > 0.35
- **Word Error Rate (WER)** < 0.40

Execute the evaluation command (adjust paths if custom splits are used):
```bash
python src/evaluate.py --phase 2 --config config_phase2.yaml --split val
```

### Mandatory Qualitative Spot-Check
Do not trust metric scores alone. Look at the output of the evaluation run (which prints sample targets and predicted sentences) and manually spot-check **10 to 15 predictions**:
1. Check for telegraphic word drops (ensure helper words like copulas and articles are outputted where appropriate).
2. Check for alignment of actions and subjects (confirm names/nouns are not hallucinated or swapped).
3. Ensure no repetitive loops are generated.

---

## 5. Architectural Assumptions & Hooks Checklist

The training script interfaces with the model classes defined in your existing `src/model.py` and `src/dataset.py`. If you customized method names, verify that they match these contracts:

*   **`ASLTranslationModel` constructor** (`src/model.py`):
    ```python
    model = ASLTranslationModel(
        t5_model_name=cfg["t5_model"],
        landmark_dim=cfg["landmark_dim"],
        projector_hidden=cfg["projector_hidden"],
        t5_hidden=cfg["t5_hidden_size"],
        max_frames=cfg["max_frames"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"]
    )
    ```
*   **Model placement**: `model.projector.to(device)` is used to place the sequence encoder. T5 is distributed automatically on initialization via the `device_map="auto"` argument inside PEFT.
*   **Gradient checkpointing**: `model.t5.gradient_checkpointing_enable()` is called to enable memory-efficient backward passes.
*   **Parameter grouping**: `model.trainable_parameter_groups(lr_projector, lr_lora)` is called to set separate learning rates for the projector and LoRA parameters.
*   **Checkpoint loading/saving hooks**:
    - Projector states are saved/loaded via `model.projector.state_dict()`.
    - LoRA adapter parameters are saved via `model.t5.save_pretrained(path)` and loaded via `model.t5.load_adapter(path, adapter_name="default", is_trainable=True)`.
*   **Input Gradients Activation**: `model.enable_input_require_grads()` is called on the PEFT/T5 wrapper at instantiation to ensure gradients flow back to the trainable projector.
