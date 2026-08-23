"""
emotion_recognition.py — Phase Aug-Sep

Facial emotion recognition from MediaPipe facial landmarks (468 points → 3D)
or raw face crops using DeepFace / FER as backends.

Pipeline:
    1. Receive a sequence of facial landmark frames (T, 468, 3) extracted by
       extract_landmarks.py (the face sub-array of the 345-dim holistic output).
    2. Optionally fall back to DeepFace on raw BGR face crops when landmark
       confidence is low.
    3. Return a per-clip emotion label and a confidence score dict.

Planned emotions: angry, disgust, fear, happy, sad, surprise, neutral

TODO (Aug-Sep):
    - Implement landmark-to-emotion MLP (lightweight, CPU-friendly).
    - Integrate DeepFace fallback for low-confidence frames.
    - Add temporal smoothing (rolling majority vote over ±3 frames).
    - Unit tests with synthetic landmark arrays.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EmotionRecognizer:
    """Placeholder — not yet implemented."""

    def __init__(self, backend: str = "landmark_mlp", device: str = "cpu"):
        """
        Args:
            backend: 'landmark_mlp' | 'deepface' | 'fer'
            device:  'cpu' | 'cuda'
        """
        self.backend = backend
        self.device = device
        # TODO: load trained MLP weights here
        raise NotImplementedError(
            "EmotionRecognizer is scheduled for Aug-Sep development. "
            "Set backend='deepface' and install `pip install deepface` "
            "for an interim solution."
        )

    def predict(
        self,
        face_landmarks: np.ndarray,          # shape (T, 468, 3)
        face_crops: Optional[np.ndarray] = None,  # shape (T, H, W, 3) BGR
    ) -> dict:
        """
        Returns:
            {
                'label': str,           # dominant emotion
                'scores': dict[str, float],  # per-emotion softmax scores
                'confidence': float,
            }
        """
        raise NotImplementedError
