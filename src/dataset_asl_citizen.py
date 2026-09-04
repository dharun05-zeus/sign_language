"""
PyTorch Dataset for ASL Citizen signer diversity training (Phase 3).
Loads 345-dimensional landmark files and parses ASL Citizen CSV metadata,
supporting top-100 most frequent class filtering and in-memory preloading.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ASLCitizenDataset(Dataset):
    def __init__(
        self,
        metadata_csv="data/asl_citizen/splits/train.csv",
        landmarks_dir="data/landmarks/asl_citizen",
        split=None,
        top_n_classes=100,
        max_frames=150,
        in_memory=True,
    ):
        """
        Args:
            metadata_csv (str): Path to ASL Citizen split CSV (train.csv, val.csv, or splits.csv).
            landmarks_dir (str): Directory containing pre-extracted 345-dim landmark .npy files.
            split (str, optional): Target split to filter by if using a combined splits.csv ('train', 'val', 'test').
            top_n_classes (int, optional): Filter for top N most frequent sign classes (default: 100).
            max_frames (int): Sequence length cutoff (150 frames).
            in_memory (bool): Preload numpy arrays into RAM for zero-latency batch iteration.
        """
        self.max_frames = max_frames
        self.in_memory = in_memory
        self.landmarks_dir = landmarks_dir

        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"ASL Citizen metadata CSV not found at: {metadata_csv}")

        df = pd.read_csv(metadata_csv)

        # Standardize column names (lowercase with no spaces)
        col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)

        # Resolve video file column
        video_col = next((c for c in ["video_file", "filename", "video_id", "file", "clip_id"] if c in df.columns), df.columns[0])
        
        # Resolve gloss / text column
        text_col = next((c for c in ["gloss", "text", "label", "sign", "clean_text"] if c in df.columns), df.columns[1])
        
        # Resolve split column if present
        if split is not None and "split" in df.columns:
            df = df[df["split"].str.lower() == split.lower()].reset_index(drop=True)

        # Top-N Most Frequent Class Filtering
        if top_n_classes is not None and top_n_classes > 0:
            top_classes = df[text_col].value_counts().head(top_n_classes).index.tolist()
            df = df[df[text_col].isin(top_classes)].reset_index(drop=True)
            print(f"Filtered to top-{top_n_classes} sign classes ({len(df)} samples remaining).")

        self.samples = []
        for _, row in df.iterrows():
            raw_video = str(row[video_col])
            base_name = os.path.splitext(os.path.basename(raw_video))[0]
            
            # Possible landmark file locations
            possible_paths = [
                os.path.join(landmarks_dir, f"{base_name}.npy"),
                os.path.join(landmarks_dir, f"{raw_video}.npy"),
                os.path.join(landmarks_dir, raw_video),
            ]
            
            resolved_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])
            
            self.samples.append({
                "landmark_path": resolved_path,
                "text": str(row[text_col]).strip().lower(),
                "signer_id": str(row.get("signer_id", row.get("user", "unknown"))),
            })

        print(f"Loaded ASL Citizen dataset ({split or 'all'}): {len(self.samples)} samples across "
              f"{len(set(s['text'] for s in self.samples))} unique sign classes.")

        self.cached_landmarks = []
        self.cached_attention_masks = []

        if self.in_memory and len(self.samples) > 0:
            print(f"Preloading ASL Citizen landmarks to RAM (split={split})...")
            for idx in tqdm(range(len(self.samples)), desc="Preloading ASL Citizen"):
                landmarks, attention_mask = self._load_and_process(idx)
                self.cached_landmarks.append(landmarks)
                self.cached_attention_masks.append(attention_mask)

    def _load_and_process(self, idx):
        row = self.samples[idx]
        landmarks_path = row["landmark_path"]

        if not os.path.exists(landmarks_path):
            landmarks = np.zeros((self.max_frames, 345), dtype=np.float32)
            attention_mask = np.zeros(self.max_frames, dtype=np.float32)
            return landmarks, attention_mask

        landmarks = np.load(landmarks_path).astype(np.float32)  # (T, 345)
        t = landmarks.shape[0]

        if t < self.max_frames:
            pad = np.zeros((self.max_frames - t, landmarks.shape[1]), dtype=np.float32)
            attention_mask = np.concatenate([
                np.ones(t, dtype=np.float32),
                np.zeros(self.max_frames - t, dtype=np.float32)
            ])
            landmarks = np.concatenate([landmarks, pad], axis=0)
        else:
            landmarks = landmarks[:self.max_frames]
            attention_mask = np.ones(self.max_frames, dtype=np.float32)

        return landmarks, attention_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.in_memory and len(self.cached_landmarks) == len(self.samples):
            landmarks = self.cached_landmarks[idx]
            attention_mask = self.cached_attention_masks[idx]
        else:
            landmarks, attention_mask = self._load_and_process(idx)

        text = str(self.samples[idx]["text"])
        return {
            "landmarks": torch.from_numpy(landmarks),
            "attention_mask": torch.from_numpy(attention_mask),
            "text": text,
        }


def collate_fn(batch):
    landmarks = torch.stack([b["landmarks"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    return {
        "landmarks": landmarks,
        "attention_mask": attention_mask,
        "texts": texts,
    }
