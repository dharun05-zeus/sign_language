"""
Builds How2Sign index manifests and transcripts from raw .npy files and official TSV transcripts.
Cross-references existing landmark files on disk with translation sentences.

Usage:
    python scripts/build_how2sign_index.py \
        --npy_dir data/landmarks/how2sign/train \
        --tsv_path data/how2sign/how2sign_realign_train.tsv \
        --split train \
        --out_manifest data/how2sign/train_manifest.csv \
        --out_transcripts data/how2sign/train_transcripts.tsv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build How2Sign Index Manifest and Transcripts")
    parser.add_argument("--npy_dir", required=True, help="Directory containing the extracted How2Sign .npy landmark files")
    parser.add_argument("--tsv_path", required=True, help="Path to the official How2Sign TSV file (e.g. how2sign_realign_train.tsv)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split (train/val/test)")
    parser.add_argument("--out_manifest", required=True, help="Output path for the generated manifest CSV")
    parser.add_argument("--out_transcripts", required=True, help="Output path for the generated transcripts TSV")
    args = parser.parse_args()

    # 1. Verify inputs
    if not os.path.isdir(args.npy_dir):
        raise NotADirectoryError(f"Landmarks directory not found: {args.npy_dir}")
    if not os.path.exists(args.tsv_path):
        raise FileNotFoundError(f"Transcripts TSV file not found: {args.tsv_path}")

    # Create output directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_transcripts)), exist_ok=True)

    # 2. Scan the .npy directory
    print(f"Scanning directory: {args.npy_dir} for .npy landmark files...")
    all_files = os.listdir(args.npy_dir)
    npy_files = {os.path.splitext(f)[0]: os.path.join(args.npy_dir, f) for f in all_files if f.endswith(".npy")}
    print(f"Found {len(npy_files)} landmark files on disk.")

    if len(npy_files) == 0:
        print("WARNING: No .npy files found in the specified directory. Double-check the path.")

    # 3. Load the official How2Sign TSV file
    print(f"Loading transcripts TSV: {args.tsv_path}...")
    df_tsv = pd.read_csv(args.tsv_path, sep="\t")

    # Column identification (support different variations of the How2Sign TSV format)
    # The official format has 'sentence_id' (matching npy filename) and 'clean_text' (translation sentence)
    id_col = "sentence_id" if "sentence_id" in df_tsv.columns else (
        "clip_id" if "clip_id" in df_tsv.columns else df_tsv.columns[0]
    )
    text_col = "clean_text" if "clean_text" in df_tsv.columns else (
        "sentence" if "sentence" in df_tsv.columns else (
            "utterance" if "utterance" in df_tsv.columns else df_tsv.columns[-1]
        )
    )
    
    print(f"Using TSV column '{id_col}' as ID and '{text_col}' as the translation sentence.")

    manifest_rows = []
    transcript_rows = []
    matched_count = 0
    missing_count = 0

    # 4. Map existing files on disk to TSV entries
    for _, row in df_tsv.iterrows():
        clip_id = str(row[id_col]).strip()
        sentence = str(row[text_col]).strip()

        if clip_id in npy_files:
            # We have a matching file on disk
            manifest_rows.append({
                "video_id": clip_id,
                "landmark_path": npy_files[clip_id].replace("\\", "/"),  # Normalize paths for Windows/Linux
                "split": args.split
            })
            transcript_rows.append({
                "clip_id": clip_id,
                "sentence": sentence
            })
            matched_count += 1
        else:
            missing_count += 1

    print(f"Processed TSV rows: Matched={matched_count} | Not found on disk={missing_count}")

    if matched_count == 0:
        print("ERROR: Zero matches found. Please verify that your .npy filenames correspond to the IDs in the TSV.")
        return

    # 5. Write outputs
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(args.out_manifest, index=False)
    print(f"Wrote manifest CSV (containing file paths) to: {args.out_manifest}")

    df_trans = pd.DataFrame(transcript_rows)
    df_trans.to_csv(args.out_transcripts, sep="\t", index=False)
    print(f"Wrote transcripts TSV (containing translations) to: {args.out_transcripts}")


if __name__ == "__main__":
    main()
