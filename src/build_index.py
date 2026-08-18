"""
Build a filtered index CSV for a WLASL subset (e.g. WLASL100) from the
pre-built nslt_{subset}.json split file, cross-referenced against
wlasl_class_list.txt for gloss names.

Usage:
    python build_index.py --wlasl_root "C:\\path\\to\\wlasl-complete" --subset 100

Expects this folder layout (matches the Kaggle wlasl-complete download):
    wlasl_root/
        videos/                  # {video_id}.mp4
        nslt_100.json
        nslt_300.json
        nslt_1000.json
        nslt_2000.json
        wlasl_class_list.txt
        WLASL_v0.3.json
        missing.txt

Output:
    wlasl_root/index_wlasl{subset}.csv with columns:
        video_id, video_path, class_id, gloss, frame_start, frame_end, split
"""

import argparse
import csv
import json
import os


def load_class_list(path):
    """wlasl_class_list.txt is tab-delimited: <class_id>\t<gloss> per line."""
    id_to_gloss = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"WARNING: could not parse class list line: {line!r}")
                continue
            class_id, gloss = parts
            id_to_gloss[int(class_id)] = gloss.strip()
    return id_to_gloss


def load_missing(path):
    missing = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    missing.add(vid)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wlasl_root", required=True, help="Path to wlasl-complete folder")
    parser.add_argument("--subset", default="100", choices=["100", "300", "1000", "2000"])
    args = parser.parse_args()

    root = args.wlasl_root
    nslt_path = os.path.join(root, f"nslt_{args.subset}.json")
    class_list_path = os.path.join(root, "wlasl_class_list.txt")
    videos_dir = os.path.join(root, "videos")
    missing_path = os.path.join(root, "missing.txt")
    out_path = os.path.join(root, f"index_wlasl{args.subset}.csv")

    print(f"Loading {nslt_path} ...")
    with open(nslt_path, "r", encoding="utf-8") as f:
        nslt = json.load(f)

    print(f"Loading {class_list_path} ...")
    id_to_gloss = load_class_list(class_list_path)

    missing = load_missing(missing_path)
    print(f"Loaded {len(missing)} known-missing video ids")

    rows = []
    skipped_no_file = 0
    skipped_no_gloss = 0

    for video_id, entry in nslt.items():
        subset = entry.get("subset")
        action = entry.get("action")
        if action is None or len(action) < 3:
            continue
        class_id, frame_start, frame_end = action[0], action[1], action[2]

        gloss = id_to_gloss.get(class_id)
        if gloss is None:
            skipped_no_gloss += 1
            continue

        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(videos_dir, video_filename)

        if video_id in missing or not os.path.exists(video_path):
            skipped_no_file += 1
            continue

        rows.append({
            "video_id": video_id,
            "video_path": video_path,
            "class_id": class_id,
            "gloss": gloss,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "split": subset,
        })

    print(f"Total entries in nslt_{args.subset}.json: {len(nslt)}")
    print(f"Skipped (no gloss mapping): {skipped_no_gloss}")
    print(f"Skipped (file missing/not found): {skipped_no_file}")
    print(f"Usable rows: {len(rows)}")

    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    print(f"Split breakdown: {split_counts}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "video_path", "class_id", "gloss",
                        "frame_start", "frame_end", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote index to {out_path}")


if __name__ == "__main__":
    main()
