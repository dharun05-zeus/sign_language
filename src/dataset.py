"""
PyTorch Dataset for landmark -> text training, used across all three phases.

Phase 1 (WLASL): target text is just the gloss word itself (e.g. "book").
Phase 2 (How2Sign): target text is the full English sentence transcript.
Phase 3 (MS-ASL): target text is the gloss word, same shape as Phase 1.

All phases share the same __getitem__ contract:
    {
        "landmarks": FloatTensor (max_frames, 345),
        "attention_mask": FloatTensor (max_frames,)  -- 1 for real frames, 0 for padding
        "text": str  -- target English text, tokenized by the caller (train.py)
    }

The manifest CSV (produced by extract_landmarks.py) must have a
"landmark_path" column and a "text" column. If your manifest only has
"gloss" (as build_index.py produces for WLASL), pass --text_col gloss to
train.py / dataset construction, or rename the column before training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandmarkTextDataset(Dataset):
    def __init__(self, manifest_csv, text_col="gloss", split=None, max_frames=150, in_memory=True):
        self.df = pd.read_csv(manifest_csv)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if text_col not in self.df.columns:
            raise ValueError(
                f"text_col='{text_col}' not found in manifest columns: "
                f"{list(self.df.columns)}. Pass the correct --text_col."
            )
        self.text_col = text_col
        self.max_frames = max_frames
        self.in_memory = in_memory
        
        self.cached_landmarks: list[torch.Tensor] = []
        self.cached_attention_masks: list[torch.Tensor] = []

        if self.in_memory:
            print(f"Preloading dataset into RAM as float32 tensors (split={split}, n={len(self.df)})...")
            from tqdm import tqdm
            for idx in tqdm(range(len(self.df)), desc="Preloading landmarks"):
                landmarks_np, mask_np = self._load_and_process(idx)
                # Store as CPU tensors so __getitem__ returns them directly
                # without any per-call numpy->tensor conversion overhead.
                # .clone() frees the numpy backing array reference.
                self.cached_landmarks.append(
                    torch.from_numpy(landmarks_np).clone()
                )
                self.cached_attention_masks.append(
                    torch.from_numpy(mask_np).clone()
                )

    def __len__(self):
        return len(self.df)

    def _load_and_process(self, idx):
        row = self.df.iloc[idx]
        landmarks = np.load(row["landmark_path"]).astype(np.float32)  # (T, 345)

        t = landmarks.shape[0]
        if t < self.max_frames:
            pad = np.zeros((self.max_frames - t, landmarks.shape[1]), dtype=np.float32)
            attention_mask = np.concatenate(
                [np.ones(t, dtype=np.float32), np.zeros(self.max_frames - t, dtype=np.float32)]
            )
            landmarks = np.concatenate([landmarks, pad], axis=0)
        else:
            landmarks = landmarks[: self.max_frames]
            attention_mask = np.ones(self.max_frames, dtype=np.float32)
            
        return landmarks, attention_mask

    def __getitem__(self, idx):
        if self.in_memory:
            # Cached tensors: no disk I/O, no numpy conversion
            landmarks = self.cached_landmarks[idx]
            attention_mask = self.cached_attention_masks[idx]
        else:
            landmarks_np, mask_np = self._load_and_process(idx)
            landmarks = torch.from_numpy(landmarks_np)
            attention_mask = torch.from_numpy(mask_np)

        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        return {
            "landmarks": landmarks,
            "attention_mask": attention_mask,
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
