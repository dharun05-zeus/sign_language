"""
Evaluate a trained checkpoint on a held-out split using BLEU-4, ROUGE-L, and WER.

Usage:
    python src/evaluate.py --phase 2 --config config.yaml --split test
"""

import argparse
import os
import sys

import jiwer
import torch
import yaml
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import LandmarkTextDataset, collate_fn
from model import ASLTranslationModel
from train import load_checkpoint_if_exists, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    phase_key = f"phase{args.phase}"
    phase_cfg = cfg[phase_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_csv = args.manifest or os.path.join(
        cfg["paths"]["landmarks_root"], phase_cfg["dataset"].lower(), "manifest.csv"
    )
    default_text_col = "gloss" if args.phase in (1, 3) else "sentence"
    text_col = args.text_col or default_text_col

    ds = LandmarkTextDataset(manifest_csv, text_col=text_col, split=args.split,
                              max_frames=cfg["max_frames"])
    print(f"Evaluating on {len(ds)} examples from split='{args.split}'")
    if len(ds) == 0:
        print("ERROR: no examples found for this split.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ASLTranslationModel(
        t5_model_name=cfg["t5_model"],
        landmark_dim=cfg["landmark_dim"],
        projector_hidden=cfg["projector_hidden"],
        t5_hidden=cfg["t5_hidden_size"],
        max_frames=cfg["max_frames"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    )
    ckpt_dir = phase_cfg["out_dir"]
    model = load_checkpoint_if_exists(model, ckpt_dir, device)
    model.projector.to(device)
    model.eval()

    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            landmarks = batch["landmarks"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["texts"]

            max_new_tokens = 128 if args.phase == 2 else 32
            preds = model.generate(
                landmarks, attention_mask, max_new_tokens=max_new_tokens, num_beams=args.num_beams
            )
            predictions.extend(preds)
            references.extend(texts)

    bleu = load_metric("sacrebleu")
    rouge = load_metric("rouge")

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_score = rouge.compute(predictions=predictions, references=references)

    # jiwer expects non-empty strings; guard against empty predictions
    safe_preds = [p if p.strip() else "<empty>" for p in predictions]
    safe_refs = [r if r.strip() else "<empty>" for r in references]
    wer_score = jiwer.wer(safe_refs, safe_preds)

    print("\n===== Evaluation Results =====")
    print(f"BLEU-4 : {bleu_score['score']:.2f}  (target > 10, SOTA on How2Sign = 12.39)")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}  (target > 0.35)")
    print(f"WER    : {wer_score:.4f}  (target < 0.4)")
    print("===============================\n")

    print("Sample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"  REF : {references[i]}")
        print(f"  PRED: {predictions[i]}")
        print()


if __name__ == "__main__":
    main()
