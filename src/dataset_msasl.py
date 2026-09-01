"""
PyTorch Dataset for MS-ASL signer diversity training (Phase 3).
Loads 345-dimensional landmark files and parses MS-ASL JSON metadata,
synonym groupings, and class labels.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_msasl_synonyms(synonym_path):
    """
    Loads MSASL_synonym.json which maps synonymous sign names / class variants
    to a canonical gloss representation.
    """
    if not synonym_path or not os.path.exists(synonym_path):
        return {}
    
    with open(synonym_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle dict or list formats commonly found in MSASL distributions
    synonym_map = {}
    if isinstance(data, dict):
        for k, v in data.items():
            synonym_map[str(k).strip().lower()] = str(v).strip().lower()
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "synonym" in entry and "canonical" in entry:
                synonym_map[str(entry["synonym"]).strip().lower()] = str(entry["canonical"]).strip().lower()
            elif isinstance(entry, list) and len(entry) >= 2:
                canonical = str(entry[0]).strip().lower()
                for syn in entry:
                    synonym_map[str(syn).strip().lower()] = canonical
    return synonym_map


def load_msasl_classes(classes_path):
    """
    Loads MSASL_classes.json containing class name strings.
    """
    if not classes_path or not os.path.exists(classes_path):
        return []
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return [str(c).strip() for c in classes]


class MSASLDataset(Dataset):
    def __init__(
        self,
        json_path=None,
        landmarks_dir="data/landmarks/msasl100",
        classes_path="data/msasl/MSASL_classes.json",
        synonym_path="data/msasl/MSASL_synonym.json",
        manifest_csv=None,
        split=None,
        max_classes=100,
        max_frames=150,
        in_memory=True,
    ):
        """
        Args:
            json_path (str, optional): Path to MSASL_{split}.json.
            landmarks_dir (str): Directory containing pre-extracted .npy landmark files (345-dim).
            classes_path (str, optional): Path to MSASL_classes.json.
            synonym_path (str, optional): Path to MSASL_synonym.json.
            manifest_csv (str, optional): If pre-built into manifest CSV, loads directly from CSV.
            split (str, optional): Split name ('train', 'val', or 'test').
            max_classes (int): Subset cutoff (100 for MS-ASL100).
            max_frames (int): Target frame sequence length.
            in_memory (bool): If True, preloads arrays to RAM for high-throughput GPU training.
        """
        self.max_frames = max_frames
        self.in_memory = in_memory
        self.max_classes = max_classes
        self.landmarks_dir = landmarks_dir
        
        self.classes = load_msasl_classes(classes_path)
        self.synonyms = load_msasl_synonyms(synonym_path)
        
        self.samples = []
        
        # Mode A: Load from pre-indexed manifest CSV if available
        if manifest_csv and os.path.exists(manifest_csv):
            print(f"Loading MS-ASL dataset from manifest CSV: {manifest_csv}")
            df = pd.read_csv(manifest_csv)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split].reset_index(drop=True)
            if "class_id" in df.columns and self.max_classes is not None:
                df = df[df["class_id"] < self.max_classes].reset_index(drop=True)
            
            for _, row in df.iterrows():
                gloss = str(row.get("text", row.get("gloss", "")))
                canon_gloss = self.synonyms.get(gloss.strip().lower(), gloss)
                self.samples.append({
                    "landmark_path": row["landmark_path"],
                    "text": canon_gloss,
                    "class_id": int(row.get("class_id", -1)),
                    "signer_id": str(row.get("signer_id", "unknown")),
                })
        
        # Mode B: Parse official MS-ASL JSON metadata file
        elif json_path and os.path.exists(json_path):
            print(f"Parsing MS-ASL JSON split file: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                raw_entries = json.load(f)
            
            for entry in raw_entries:
                label = int(entry.get("label", -1))
                if self.max_classes is not None and label >= self.max_classes:
                    continue  # Filter for MS-ASL100
                
                raw_text = entry.get("clean_text") or entry.get("org_text")
                if not raw_text and label >= 0 and label < len(self.classes):
                    raw_text = self.classes[label]
                raw_text = str(raw_text or "").strip()
                
                # Apply synonym normalization
                canon_gloss = self.synonyms.get(raw_text.lower(), raw_text)
                
                # Locate landmark .npy file
                file_id = entry.get("file") or f"{entry.get('video_id', '')}_{entry.get('start_time', '')}_{entry.get('end_time', '')}"
                # Clean filename
                file_base = os.path.splitext(os.path.basename(str(file_id)))[0]
                possible_paths = [
                    os.path.join(landmarks_dir, f"{file_base}.npy"),
                    os.path.join(landmarks_dir, f"{entry.get('video_id', '')}.npy"),
                    os.path.join(landmarks_dir, str(entry.get("file", ""))),
                ]
                
                resolved_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        resolved_path = p
                        break
                
                if resolved_path is None:
                    # Default path expectation if files are being staged
                    resolved_path = possible_paths[0]
                
                self.samples.append({
                    "landmark_path": resolved_path,
                    "text": canon_gloss,
                    "class_id": label,
                    "signer_id": str(entry.get("signer_id", "unknown")),
                })
        else:
            raise FileNotFoundError(
                f"Neither valid manifest_csv ({manifest_csv}) nor json_path ({json_path}) was found."
            )

        print(f"Loaded MS-ASL100 dataset ({split or 'all'}): {len(self.samples)} valid samples.")

        self.cached_landmarks = []
        self.cached_attention_masks = []

        if self.in_memory and len(self.samples) > 0:
            print(f"Preloading MS-ASL100 landmarks to RAM (split={split})...")
            loaded_count = 0
            for idx in tqdm(range(len(self.samples)), desc="Preloading MS-ASL100"):
                landmarks, attention_mask = self._load_and_process(idx)
                self.cached_landmarks.append(landmarks)
                self.cached_attention_masks.append(attention_mask)
                loaded_count += 1

    def _load_and_process(self, idx):
        row = self.samples[idx]
        landmarks_path = row["landmark_path"]
        
        if not os.path.exists(landmarks_path):
            # Fallback zero-filled array of shape (max_frames, 345) if clip file is missing
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
