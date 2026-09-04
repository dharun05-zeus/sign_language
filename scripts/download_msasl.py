"""
download_msasl.py — Robust MS-ASL video dataset downloader using yt-dlp.

Fixes filename collisions by extracting unique YouTube video IDs and timestamp ranges
from the official MS-ASL JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
                   help="Parallel download workers (default: 2 to avoid YouTube rate limits)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats without downloading")
    return p.parse_args()


def get_clip_id(entry: dict, idx: int = 0) -> str:
    """Derives a unique, safe filename for an MS-ASL JSON entry."""
    url = entry.get("url", "")
    
    # Extract YouTube video ID from URL
    video_id = ""
    if "v=" in url:
        video_id = url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    
    if not video_id:
        raw_file = str(entry.get("file", f"clip_{idx}"))
        video_id = re.sub(r"[^\w\-]", "_", raw_file)
        
    start = float(entry.get("start_time", entry.get("start", 0.0)))
    end = float(entry.get("end_time", entry.get("end", 0.0)))
    
    start_str = f"{start:.2f}".replace(".", "_")
    end_str = f"{end:.2f}".replace(".", "_")
    
    return f"{video_id}_{start_str}_{end_str}"


def download_clip(entry: dict, out_dir: str, idx: int) -> str:
    """
    Download a single clip using yt-dlp.
    Returns 'ok', 'skip' (already exists), or 'fail'.
    """
    url = entry.get("url", "")
    if not url:
        return "fail"

    clip_id = get_clip_id(entry, idx)
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return "skip"

    start = float(entry.get("start_time", 0.0))
    end = float(entry.get("end_time", 0.0))

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-check-certificates",
        "--retries", "3",
        "--socket-timeout", "30",
        "-f", "mp4[height<=480]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--output", out_path,
    ]

    # Download exact timestamp section if timestamps are present
    if end > start and (end - start) > 0.1:
        cmd += ["--download-sections", f"*{start:.2f}-{end:.2f}", "--force-keyframes-at-cuts"]

    cmd.append(url)

    try:
        result = subprocess.run(cmd, timeout=90, capture_output=True)
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return "ok"
        return "fail"
    except Exception:
        return "fail"


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to top-N classes (e.g. 100 for MSASL100)
    entries = [e for e in data if e.get("label", 9999) < args.subset]
    print(f"Entries for top-{args.subset} classes: {len(entries)}")

    if args.dry_run:
        print(f"[dry_run] Found {len(entries)} valid entries. No files downloaded.")
        return

    counts = {"ok": 0, "skip": 0, "fail": 0}
    bar = tqdm(total=len(entries), unit="clip") if _TQDM else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, e, args.out_dir, i): i for i, e in enumerate(entries)
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result] += 1
            if bar:
                bar.set_postfix(**counts)
                bar.update(1)

    if bar:
        bar.close()

    print(f"\nDone. Downloaded: {counts['ok']} | Already Existed: {counts['skip']} | Failed: {counts['fail']}")


if __name__ == "__main__":
    main()
