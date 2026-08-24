"""
Extract MediaPipe Holistic landmarks from video clips listed in an index CSV
(produced by scripts/build_index.py) and save them as .npy files.

Supports Windows/Linux multiprocessing and frame downsampling/filtering for 10x+ speedup.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global worker-specific MediaPipe instance
worker_holistic = None

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


def process_video(video_path, holistic, frame_start=None, frame_end=None, max_frames=150):
    """Run MediaPipe Holistic over a video (or a frame range within it).

    frame_start / frame_end are 1-indexed inclusive.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_start is None or pd.isna(frame_start):
        start_idx = 0
    else:
        start_idx = max(0, int(frame_start) - 1)

    if frame_end is None or pd.isna(frame_end) or int(frame_end) == -1:
        end_idx = total_frames - 1
    else:
        end_idx = min(total_frames - 1, int(frame_end) - 1)

    if end_idx < start_idx:
        end_idx = start_idx

    segment_len = end_idx - start_idx + 1

    # Opt 1: Downsample target frames beforehand to avoid running MediaPipe on redundant frames
    if segment_len > max_frames:
        target_indices = set(np.linspace(start_idx, end_idx, max_frames).astype(int))
    else:
        target_indices = set(range(start_idx, end_idx + 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames_landmarks = []
    current_idx = start_idx
    while current_idx <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_indices:
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


def init_worker(min_detection_confidence, min_tracking_confidence):
    global worker_holistic
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    worker_holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def worker_task(args_tuple):
    row_dict, out_dir, id_col, video_col, frame_col_start, frame_col_end, max_frames, skip_existing = args_tuple
    video_id = str(row_dict[id_col])
    video_path = row_dict[video_col]
    out_path = os.path.join(out_dir, f"{video_id}.npy")

    if skip_existing and os.path.exists(out_path):
        return {"status": "skipped", "row": row_dict, "path": out_path}

    if not os.path.exists(video_path):
        return {"status": "failed", "video_id": video_id}

    frame_start = row_dict.get(frame_col_start) if frame_col_start else None
    frame_end = row_dict.get(frame_col_end) if frame_col_end else None

    try:
        landmarks = process_video(video_path, worker_holistic, frame_start, frame_end, max_frames)
        if landmarks is None:
            return {"status": "failed", "video_id": video_id}

        landmarks = pad_or_truncate(landmarks, max_frames)
        np.save(out_path, landmarks.astype(np.float32))
        return {"status": "success", "row": row_dict, "path": out_path}
    except Exception as e:
        sys.stderr.write(f"Error processing video_id {video_id}: {e}\n")
        return {"status": "failed", "video_id": video_id}


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
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel processes (defaults to number of CPU cores minus 2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.index, dtype={args.id_col: str})
    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded index with {len(df)} rows from {args.index}")

    # Determine workers count
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 1
        args.num_workers = max(1, cpu_count - 2)

    print(f"Running landmark extraction with {args.num_workers} parallel workers...")

    # Build tasks input arguments
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            row.to_dict(),
            args.out_dir,
            args.id_col,
            args.video_col,
            args.frame_col_start,
            args.frame_col_end,
            args.max_frames,
            args.skip_existing
        ))

    manifest_rows = []
    n_ok, n_fail, n_skipped_existing = 0, 0, 0

    # Execute tasks using ProcessPoolExecutor
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args.min_detection_confidence, args.min_tracking_confidence)
    ) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting landmarks"):
            res = future.result()
            status = res["status"]
            if status == "success":
                row_dict = res["row"]
                row_dict["landmark_path"] = res["path"]
                manifest_rows.append(row_dict)
                n_ok += 1
            elif status == "skipped":
                row_dict = res["row"]
                row_dict["landmark_path"] = res["path"]
                manifest_rows.append(row_dict)
                n_skipped_existing += 1
            else:
                n_fail += 1

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\nDone. Extracted: {n_ok} | Skipped (already existed): {n_skipped_existing} | Failed: {n_fail}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
