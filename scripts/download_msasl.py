"""
download_msasl.py — MS-ASL dataset downloader helper

MS-ASL (Microsoft American Sign Language Dataset) does not have a direct
public download URL; access is granted via a research request form. This
script automates post-approval steps:

    1. Validate that the user has placed the MS-ASL JSON manifest at
       data/msasl/MSASL_*.json (downloaded after approval).
    2. Download video clips listed in the manifest using yt-dlp (YouTube).
    3. Filter to MSASL100 (top-100 classes) by default.
    4. Save downloaded .mp4 files to data/msasl/videos/.

Usage:
    python scripts/download_msasl.py \\
        --manifest data/msasl/MSASL_train.json \\
        --out_dir  data/msasl/videos \\
        --subset   100 \\
        --workers  4

Requirements:
    pip install yt-dlp tqdm

Notes:
    - Many MS-ASL YouTube links are geo-restricted or removed. Expect ~15-30%
      missing even on a fresh download.
    - Use --dry_run to count available links before committing to a full
      download.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MS-ASL video clips via yt-dlp.")
    p.add_argument("--manifest", required=True,
                   help="Path to MSASL_train.json / MSASL_val.json / MSASL_test.json")
    p.add_argument("--out_dir", default="data/msasl/videos",
                   help="Directory to save downloaded .mp4 files")
    p.add_argument("--subset", type=int, default=100,
                   help="Limit to top-N classes (default: 100 for MSASL100)")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel download workers (keep low to avoid rate-limits)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats without downloading")
    return p.parse_args()


def download_clip(entry: dict, out_dir: str) -> str:
    """
    Download a single clip using yt-dlp.
    Returns 'ok', 'skip' (already exists), or 'fail'.
    """
    url = entry.get("url", "")
    clip_id = entry.get("clip_id", entry.get("id", "unknown"))
    start = entry.get("start_time", 0)
    end = entry.get("end_time", 0)
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")

    if os.path.exists(out_path):
        return "skip"

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f", "mp4/bestvideo+bestaudio/best",
        "--output", out_path,
        url,
    ]
    # yt-dlp supports --download-sections for trimming if timestamps are provided
    if start and end:
        cmd += ["--download-sections", f"*{start:.2f}-{end:.2f}"]

    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)
        return "ok" if result.returncode == 0 else "fail"
    except Exception:
        return "fail"


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to top-N classes
    entries = [e for e in data if e.get("label", 9999) < args.subset]
    print(f"Entries for top-{args.subset} classes: {len(entries)}")

    if args.dry_run:
        print("[dry_run] No files downloaded.")
        return

    counts = {"ok": 0, "skip": 0, "fail": 0}
    bar = tqdm(total=len(entries), unit="clip") if _TQDM else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, e, args.out_dir): e for e in entries
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result] += 1
            if bar:
                bar.set_postfix(**counts)
                bar.update(1)

    if bar:
        bar.close()

    print(f"\nDone. Downloaded: {counts['ok']}  Skipped: {counts['skip']}  Failed: {counts['fail']}")


if __name__ == "__main__":
    main()
