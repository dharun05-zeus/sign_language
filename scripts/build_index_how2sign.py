import os
import pandas as pd
import numpy as np

# Load realigned train csv
metadata_path = "data/landmarks/how2sign/how2sign_train.csv"
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")

df = pd.read_csv(metadata_path, sep="\t")

# Check raw videos directory
raw_video_dir = "data/how2sign/train"
if not os.path.exists(raw_video_dir):
    raise FileNotFoundError(f"Raw videos directory not found at {raw_video_dir}")

# Map of existing files
existing_files = set(f for f in os.listdir(raw_video_dir) if f.endswith(".mp4"))
print(f"Total video files in how2sign/train: {len(existing_files)}")

# Collect matched rows
rows = []
for idx, row in df.iterrows():
    sentence_name = str(row["SENTENCE_NAME"])
    video_filename = sentence_name + ".mp4"
    if video_filename in existing_files:
        rows.append({
            "video_id": sentence_name,
            "video_path": os.path.join(raw_video_dir, video_filename).replace("\\", "/"),
            "sentence": row["SENTENCE"]
        })

matched_df = pd.DataFrame(rows)
print(f"Matched rows: {len(matched_df)}")

# Create deterministically shuffled train/val split (1000 for val, rest for train)
np.random.seed(42)
shuffled_indices = np.random.permutation(len(matched_df))

val_size = 1000
val_indices = set(shuffled_indices[:val_size])

splits = []
for i in range(len(matched_df)):
    if i in val_indices:
        splits.append("val")
    else:
        splits.append("train")

matched_df["split"] = splits

# Print split counts
print(matched_df["split"].value_counts())

# Save to CSV
out_path = "data/how2sign/index_how2sign.csv"
matched_df.to_csv(out_path, index=False)
print(f"Wrote How2Sign index to {out_path}")
