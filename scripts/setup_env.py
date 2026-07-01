import sys
import os

def check_numpy():
    print("--- Check 1: NumPy ---")
    try:
        import numpy as np
        version = np.__version__
        print(f"NumPy version: {version}")
        v_parts = [int(x) for x in version.split('.')[:2]]
        if v_parts[0] >= 2:
            print("[FAIL] NumPy version is 2.x or higher. MediaPipe might have compatibility issues.")
            print("Recommendation: Install numpy<2.0.0 (e.g. pip install 'numpy<2.0.0')")
            return False
        else:
            print("[PASS] NumPy check passed.")
            return True
    except ImportError:
        print("[FAIL] NumPy is not installed.")
        return False

def check_pytorch():
    print("\n--- Check 2: PyTorch & CUDA ---")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print("[PASS] PyTorch CUDA check passed.")
            return True
        else:
            print("[WARN] PyTorch CUDA check warning: CUDA is not available. PyTorch will run in CPU mode.")
            print("Note: This is expected on this Intel Iris Xe machine for local dev/data-prep.")
            print("[PASS] PyTorch (CPU fallback) check passed.")
            return True
    except ImportError:
        print("[FAIL] PyTorch is not installed.")
        return False

def check_bitsandbytes():
    print("\n--- Check 3: bitsandbytes ---")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
        # Test if it imports and registers properly
        try:
            from bitsandbytes.optim import AdamW
            print("[PASS] bitsandbytes successfully imported.")
            return True
        except Exception as e:
            print(f"[WARN] bitsandbytes CUDA module warning: {e}")
            print("Note: bitsandbytes requires CUDA for 4-bit quantization. It cannot run quantized training on CPU.")
            return True
    except ImportError:
        print("[FAIL] bitsandbytes is not installed.")
        return False

def check_mediapipe():
    print("\n--- Check 4: MediaPipe Holistic ---")
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[WARN] OpenCV is not installed.")
    
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
        # Test initialization
        try:
            holistic = mp.solutions.holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                refine_face_landmarks=True
            )
            holistic.close()
            print("[PASS] MediaPipe Holistic initialized successfully.")
            return True
        except Exception as e:
            print(f"[FAIL] MediaPipe Holistic initialization failed: {e}")
            return False
    except ImportError:
        print("[FAIL] MediaPipe is not installed.")
        return False

def main():
    print("ASL Pipeline Diagnostic Environment Setup Tool")
    print(f"Python version: {sys.version}\n")
    
    np_ok = check_numpy()
    torch_ok = check_pytorch()
    bnb_ok = check_bitsandbytes()
    mp_ok = check_mediapipe()
    
    print("\n================ SUMMARY ================")
    print(f"NumPy Check:       {'PASS' if np_ok else 'FAIL'}")
    print(f"PyTorch Check:     {'PASS' if torch_ok else 'FAIL'}")
    print(f"bitsandbytes:      {'PASS' if bnb_ok else 'FAIL'}")
    print(f"MediaPipe Check:   {'PASS' if mp_ok else 'FAIL'}")
    print("=========================================")
    
    if np_ok and torch_ok and bnb_ok and mp_ok:
        print("[SUCCESS] Environment diagnostic completed: READY for data preparation.")
        sys.exit(0)
    else:
        print("[FAIL] Environment diagnostic completed: Errors detected. Please fix before running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
