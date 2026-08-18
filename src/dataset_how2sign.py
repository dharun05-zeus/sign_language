"""
PyTorch Dataset for How2Sign sentence-level landmark -> English translation training.
Loads 345-dimensional landmark files and joins manifest metadata with translation sentences.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class How2SignDataset(Dataset):
    def __init__(self, manifest_csv, transcripts_tsv, split=None, max_frames=150, in_memory=True):
        """
        Args:
            manifest_csv (str): Path to How2Sign manifest CSV containing video IDs and landmark paths.
            transcripts_tsv (str): Path to How2Sign transcripts TSV containing clip/sentence IDs and text translations.
            split (str, optional): Target split to filter by (e.g. 'train', 'val', 'test').
            max_frames (int): Maximum frame sequence length to pad/truncate to.
            in_memory (bool): If True, preloads all landmark arrays into RAM to avoid disk I/O bottlenecks.
        """
        self.max_frames = max_frames
        self.in_memory = in_memory

        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"How2Sign manifest not found at: {manifest_csv}")
        if not os.path.exists(transcripts_tsv):
            raise FileNotFoundError(f"How2Sign transcripts tsv not found at: {transcripts_tsv}")

        # Load manifest
        df_manifest = pd.read_csv(manifest_csv)
        if split is not None and "split" in df_manifest.columns:
            df_manifest = df_manifest[df_manifest["split"] == split].reset_index(drop=True)

        # Load transcripts (How2Sign is standard tab-separated)
        df_trans = pd.read_csv(transcripts_tsv, sep="\t")

        # Determine joining columns (support both standard and flexible layouts)
        manifest_id_col = "video_id" if "video_id" in df_manifest.columns else "clip_id"
        trans_id_col = "clip_id" if "clip_id" in df_trans.columns else (
            "sentence_id" if "sentence_id" in df_trans.columns else df_trans.columns[0]
        )

        df_manifest[manifest_id_col] = df_manifest[manifest_id_col].astype(str)
        df_trans[trans_id_col] = df_trans[trans_id_col].astype(str)

        # Merge manifest and transcripts on clip ID
        self.df = pd.merge(df_manifest, df_trans, left_on=manifest_id_col, right_on=trans_id_col, how="inner")
        
        # Locate the translation text column
        self.text_col = "sentence" if "sentence" in self.df.columns else (
            "utterance" if "utterance" in self.df.columns else self.df.columns[-1]
        )

        print(f"Loaded How2Sign dataset ({split}): {len(self.df)} rows merged successfully from "
              f"{len(df_manifest)} manifest entries.")

        self.cached_landmarks = []
        self.cached_attention_masks = []

        if self.in_memory:
            print(f"Preloading How2Sign landmarks to RAM (split={split})...")
            for idx in tqdm(range(len(self.df)), desc="Preloading How2Sign"):
                landmarks, attention_mask = self._load_and_process(idx)
                self.cached_landmarks.append(landmarks)
                self.cached_attention_masks.append(attention_mask)

    def _load_and_process(self, idx):
        row = self.df.iloc[idx]
        landmarks_path = row["landmark_path"]
        
        if not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Landmark file not found at: {landmarks_path}")

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
        return len(self.df)

    def __getitem__(self, idx):
        if self.in_memory:
            landmarks = self.cached_landmarks[idx]
            attention_mask = self.cached_attention_masks[idx]
        else:
            landmarks, attention_mask = self._load_and_process(idx)

        row = self.df.iloc[idx]
        text = str(row[self.text_col])

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
