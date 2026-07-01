"""
Extract MediaPipe Holistic landmarks from video clips listed in an index CSV
(produced by scripts/build_index.py) and save them as .npy files.

Runs entirely on CPU - GPU is never touched here.

Landmark layout per frame (345 dims total):
    pose:        33 landmarks x 3 (x, y, z)         = 99
    left_hand:   21 landmarks x 3 (x, y, z)          = 63
    right_hand:  21 landmarks x 3 (x, y, z)          = 63
    face:        40 landmarks x 3 (x, y, z)          = 120 (reduced set: eyebrows + mouth contour)
If a landmark group is not detected in a frame, it is zero-filled rather
than skipped, so every frame is always exactly 345 dims.

Usage:
    python src/extract_landmarks.py ^
        --index data/wlasl/index_wlasl100.csv ^
        --out_dir data/landmarks/wlasl100 ^
        --frame_col_start frame_start --frame_col_end frame_end ^
        --max_frames 150

For datasets without trim columns (e.g. How2Sign full-clip sentences), omit
--frame_col_start/--frame_col_end and the whole video will be used.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe is not installed. Run: pip install mediapipe")
    sys.exit(1)

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 40
FACE_INDICES = [
    # Left eyebrow (10 landmarks)
    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
    # Right eyebrow (10 landmarks)
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # Lips contour (20 landmarks)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88
]
LANDMARK_DIM = (POSE_LANDMARKS * 3) + (HAND_LANDMARKS * 3) + (HAND_LANDMARKS * 3) + (FACE_LANDMARKS * 3)  # 345


def extract_frame_landmarks(results):
    """Flatten one MediaPipe Holistic frame result into a fixed 345-dim vector."""
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)

    if results.face_landmarks:
        face_all = results.face_landmarks.landmark
        face = np.array(
            [[face_all[idx].x, face_all[idx].y, face_all[idx].z] for idx in FACE_INDICES],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(FACE_LANDMARKS * 3, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand, face])  # (345,)


def process_video(video_path, holistic, frame_start=None, frame_end=None):
    """Run MediaPipe Holistic over a video (or a frame range within it).

    frame_start / frame_end are 1-indexed inclusive, matching the WLASL
    nslt_*.json convention. frame_end == -1 means "to the last frame".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_start is None:
        start_idx = 0
    else:
        start_idx = max(0, int(frame_start) - 1)

    if frame_end is None or int(frame_end) == -1:
        end_idx = total_frames - 1
    else:
        end_idx = min(total_frames - 1, int(frame_end) - 1)

    if end_idx < start_idx:
        end_idx = start_idx

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames_landmarks = []
    current_idx = start_idx
    while current_idx <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        frames_landmarks.append(extract_frame_landmarks(results))
        current_idx += 1

    cap.release()

    if len(frames_landmarks) == 0:
        return None

    return np.stack(frames_landmarks, axis=0)  # (T, 345)


def pad_or_truncate(arr, max_frames):
    """Pad with zeros or truncate (uniform sampling) to exactly max_frames."""
    t = arr.shape[0]
    if t == max_frames:
        return arr
    if t > max_frames:
        idx = np.linspace(0, t - 1, max_frames).astype(int)
        return arr[idx]
    pad = np.zeros((max_frames - t, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to index CSV from build_index.py")
    parser.add_argument("--out_dir", required=True, help="Directory to write .npy landmark files")
    parser.add_argument("--video_col", default="video_path")
    parser.add_argument("--id_col", default="video_id")
    parser.add_argument("--frame_col_start", default=None,
                         help="Column name for 1-indexed start frame (omit if full video)")
    parser.add_argument("--frame_col_end", default=None,
                         help="Column name for 1-indexed end frame, -1 = last frame")
    parser.add_argument("--max_frames", type=int, default=150)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of clips, for a quick smoke test")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                         help="Skip clips that already have a saved .npy (resume support)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.index, dtype={args.id_col: str})
    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded index with {len(df)} rows from {args.index}")

    mp_holistic = mp.solutions.holistic

    manifest_rows = []
    n_ok, n_fail, n_skipped_existing = 0, 0, 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting landmarks"):
            video_id = str(row[args.id_col])
            video_path = row[args.video_col]
            out_path = os.path.join(args.out_dir, f"{video_id}.npy")

            if args.skip_existing and os.path.exists(out_path):
                n_skipped_existing += 1
                manifest_rows.append({**row.to_dict(), "landmark_path": out_path})
                continue

            if not os.path.exists(video_path):
                n_fail += 1
                continue

            frame_start = row[args.frame_col_start] if args.frame_col_start else None
            frame_end = row[args.frame_col_end] if args.frame_col_end else None

            landmarks = process_video(video_path, holistic, frame_start, frame_end)
            if landmarks is None:
                n_fail += 1
                continue

            landmarks = pad_or_truncate(landmarks, args.max_frames)
            np.save(out_path, landmarks.astype(np.float32))

            manifest_rows.append({**row.to_dict(), "landmark_path": out_path})
            n_ok += 1

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\nDone. Extracted: {n_ok} | Skipped (already existed): {n_skipped_existing} | Failed: {n_fail}")
    print(f"Manifest written to {manifest_path}")
    if n_fail > 0:
        print("NOTE: failed clips were excluded from the manifest. This is normal for a "
              "small fraction of clips (corrupt files, 0-length trims). If failures are "
              "a large fraction of the dataset, check video paths and codec support.")


if __name__ == "__main__":
    main()
