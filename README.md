# ASL-to-English Translation Pipeline — Setup & Run Guide

Target hardware: RTX 4050 Laptop GPU (6GB VRAM), Windows native (no WSL).

---

## Step 0 — Folder layout

Place this repo somewhere, e.g. `C:\asl_pipeline\`. Then put your WLASL data
(from the Kaggle `wlasl-complete` download) under `data\wlasl\`:

```
asl_pipeline\
    data\
        wlasl\
            videos\                  <- your 21,095 .mp4 files
            WLASL_v0.3.json
            nslt_100.json
            nslt_300.json
            nslt_1000.json
            nslt_2000.json
            wlasl_class_list.txt
            missing.txt
            find_missing.py
    src\
    scripts\
    checkpoints\
    config.yaml
    requirements.txt
```

You can either move your existing `wlasl-complete` folder contents here, or
just edit `config.yaml` / pass `--wlasl_root` pointing at wherever it already
lives (e.g. `Downloads\archive (3)\wlasl-complete`).

---

## Step 1 — Install CUDA + PyTorch (do this FIRST, separately)

1. Install the NVIDIA driver for your RTX 4050 if not already installed
   (GeForce Experience or directly from nvidia.com — get the latest Game
   Ready or Studio driver, it bundles the necessary CUDA runtime components).
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install PyTorch with CUDA support — do NOT skip the `--index-url`, or pip
   will silently install a CPU-only build:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. Verify:
   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   You should see `True` and `NVIDIA GeForce RTX 4050 Laptop GPU`. If `False`,
   stop here — nothing downstream will work correctly until this is fixed.
   Common cause: driver installed but machine not rebooted, or a stale CPU
   torch install from a previous `pip install torch` — run
   `pip uninstall torch` first if you suspect that.

---

## Step 2 — Install remaining dependencies

```
pip install -r requirements.txt
```

`bitsandbytes` on Windows historically had rough edges — version `>=0.43.0`
(pinned in requirements.txt) has native Windows wheels. If you hit a
`bitsandbytes` import error mentioning a missing `.dll` or CUDA setup
function, run:
```
python -m bitsandbytes
```
which runs the library's built-in diagnostic and tells you specifically
what's missing.

---

## Step 3 — Build the WLASL100 index

```
python scripts\build_index.py --wlasl_root data\wlasl --subset 100
```

Expected output: ~2,038 total entries, 0 skipped for missing gloss mapping,
0 (or very few) skipped for missing files, with a train/val/test split
breakdown printed. This writes `data\wlasl\index_wlasl100.csv`.

If you see large numbers in "Skipped (file missing/not found)", double
check `--wlasl_root` points at the folder that directly contains `videos\`.

---

## Step 4 — Extract MediaPipe landmarks (CPU, ~1-2 hours for WLASL100)

```
python src\extract_landmarks.py ^
    --index data\wlasl\index_wlasl100.csv ^
    --out_dir data\landmarks\wlasl100 ^
    --frame_col_start frame_start --frame_col_end frame_end ^
    --max_frames 150
```

This runs entirely on CPU. It writes one `.npy` file per clip plus a
`manifest.csv` in the output dir — that manifest is what `train.py` reads.

Tip: do a quick smoke test first with `--limit 20` to catch path/codec
issues before committing to the full ~2,000-clip run:
```
python src\extract_landmarks.py --index data\wlasl\index_wlasl100.csv --out_dir data\landmarks\wlasl100_test --limit 20
```

The script supports resuming — if interrupted, re-run the same command and
it will skip clips that already have a saved `.npy`.

---

## Step 5 — Train Phase 1

```
python src\train.py --phase 1 --config config.yaml
```

This reads `data\landmarks\wlasl100\manifest.csv` (note: `train.py` expects
the manifest path to follow the pattern
`data/landmarks/{dataset_name_lowercased}/manifest.csv` by default — for
phase 1 that's `data/landmarks/wlasl100/manifest.csv`, matching Step 4's
`--out_dir` above. Override with `--manifest <path>` if you used a different
output directory).

Watch GPU memory with `nvidia-smi` (Windows: `nvidia-smi.exe`, usually in
`C:\Program Files\NVIDIA Corporation\NVSMI\`) in a second terminal during the
first few batches to confirm you're inside the ~4.4GB budget.

Checkpoints save to `checkpoints\phase1\` after every epoch, plus a `best\`
subfolder for the lowest validation loss.

---

## Step 6 — Evaluate Phase 1

```
python src\evaluate.py --phase 1 --config config.yaml --split test
```

Phase 1 is isolated-word recognition (WLASL), so BLEU-4/ROUGE-L/WER targets
in the knowledge base apply to Phase 2 (How2Sign sentences) — for Phase 1,
focus on whether predicted glosses roughly match target glosses in the
sample output printed at the end.

---

## Step 7+ — Phase 2 and Phase 3 (later)

Once Phase 1 looks reasonable:
1. Download How2Sign, extract landmarks the same way (no `frame_start`/
   `frame_end` needed — omit those flags to use the full clip), build a
   manifest with a `sentence` text column instead of `gloss`.
2. `python src\train.py --phase 2 --config config.yaml` — this automatically
   loads Phase 1's checkpoint first.
3. Repeat for Phase 3 with MS-ASL100.

---

## Step 8 — Grammar correction + serving

```
ollama pull mistral
ollama serve
```

In another terminal:
```
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Test the grammar correction layer standalone first:
```
python src\grammar_correct.py "me want water drink now please"
```

---

## Common pitfalls

- **`torch.cuda.is_available()` is False**: see Step 1. Don't proceed past
  this until fixed — 4-bit quantization and LoRA both require CUDA.
- **OOM during training**: confirm `fp16: true` and
  `gradient_checkpointing: true` in `config.yaml`, and that `batch_size: 4`
  hasn't been edited upward. Effective batch size of 16 comes from
  gradient accumulation (×4), not raw batch size.
- **MediaPipe install fails**: MediaPipe wheels lag behind the newest Python
  versions on Windows. If `pip install mediapipe` fails, check your Python
  version (`python --version`) — 3.10 or 3.11 has the broadest MediaPipe
  wheel support as of this writing.
- **`numpy` version conflicts**: `requirements.txt` pins `numpy<2.0.0`
  because MediaPipe was not numpy-2.x compatible as of the versions pinned
  here. If a newer mediapipe release has fixed this, you can relax the pin.
