import os
import numpy as np

def verify():
    test_dir = "data/landmarks/wlasl100_test"
    if not os.path.exists(test_dir):
        print(f"ERROR: test directory {test_dir} does not exist. Run smoke test first.")
        return

    npy_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]
    if not npy_files:
        print(f"ERROR: no .npy files found in {test_dir}")
        return

    print(f"Verifying facial landmark dimensions (indices 225 to 345) for {len(npy_files)} files.")
    print("=" * 80)
    print(f"{'Filename':<15} | {'Mean':<10} | {'Std':<10} | {'All-Zero Frames Fraction':<25}")
    print("-" * 80)

    for fname in npy_files:
        path = os.path.join(test_dir, fname)
        arr = np.load(path)  # shape: (150, 345)
        
        # Facial dimensions are from index 225 to 345 (120 elements)
        face_part = arr[:, 225:345]  # shape: (150, 120)
        
        mean_val = np.mean(face_part)
        std_val = np.std(face_part)
        
        # Check for each frame if all 120 values are zero
        all_zero_frames = np.all(face_part == 0, axis=1)  # shape: (150,)
        zero_fraction = np.mean(all_zero_frames)
        
        print(f"{fname:<15} | {mean_val:<10.5f} | {std_val:<10.5f} | {zero_fraction:<25.2%}")

    print("=" * 80)

if __name__ == "__main__":
    verify()
