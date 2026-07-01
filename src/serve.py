"""
FastAPI + WebSocket server for real-time ASL-to-English translation.

Pipeline per incoming message:
    landmarks (from client-side MediaPipe, or raw frames - see note below)
    -> LandmarkProjector + T5 (phase3 checkpoint, final model)
    -> Mistral 7B grammar correction (Ollama)
    -> confidence gate (threshold from config.yaml)
    -> intent classification
    -> JSON response

NOTE ON WHERE MEDIAPIPE RUNS: this server expects the CLIENT (browser/webcam
capture process) to have already run MediaPipe Holistic and to send raw
345-dim landmark frames over the WebSocket, NOT raw video. This matches the
project's design (MediaPipe runs on CPU, separate from the GPU inference
server) and avoids sending video frames over the wire. If your frontend
instead sends raw frames, run extract_landmarks.py's frame-processing logic
client-side, or add a video decode step here (not implemented - adds latency
and couples webcam decode to the GPU server process).

Run:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000

Endpoint:
    ws://localhost:8000/ws/translate
    Input:  {"landmarks": [[345 floats], [345 floats], ...]}  (T frames, T <= max_frames)
    Output: {
        "sentence": str,
        "intent": str,
        "raw_translation": str,
        "confidence": float
    }
"""

import os
import sys

import numpy as np
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ASLTranslationModel
from train import load_checkpoint_if_exists, load_config
from grammar_correct import correct_grammar
from intent_classifier import classify_intent

app = FastAPI(title="ASL-to-English Translation Pipeline")

CONFIG_PATH = os.environ.get("ASL_CONFIG_PATH", "config.yaml")
PHASE_FOR_SERVING = int(os.environ.get("ASL_SERVE_PHASE", "3"))  # final model = phase 3

_state = {}


@app.on_event("startup")
def load_model():
    cfg = load_config(CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device={device} ...")

    model = ASLTranslationModel(
        t5_model_name=cfg["t5_model"],
        landmark_dim=cfg["landmark_dim"],
        projector_hidden=cfg["projector_hidden"],
        t5_hidden=cfg["t5_hidden_size"],
        max_frames=cfg["max_frames"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    )

    ckpt_dir = cfg[f"phase{PHASE_FOR_SERVING}"]["out_dir"]
    best_dir = os.path.join(ckpt_dir, "best")
    load_dir = best_dir if os.path.exists(os.path.join(best_dir, "projector.pt")) else ckpt_dir

    model = load_checkpoint_if_exists(model, load_dir, device)
    model.projector.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device
    _state["max_frames"] = cfg["max_frames"]
    _state["landmark_dim"] = cfg["landmark_dim"]
    _state["confidence_threshold"] = cfg.get("grammar_confidence_threshold", 0.75)

    print("Model loaded. Server ready.")


def preprocess_landmarks(raw_landmarks, max_frames, landmark_dim):
    """raw_landmarks: list of T frames, each a list of `landmark_dim` floats.
    Pads/truncates to max_frames and builds an attention mask, matching the
    same convention used in dataset.py / extract_landmarks.py."""
    arr = np.array(raw_landmarks, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != landmark_dim:
        raise ValueError(
            f"Expected landmarks shaped (T, {landmark_dim}), got {arr.shape}. "
            f"Make sure the client extracts pose(33)+left_hand(21)+right_hand(21) "
            f"landmarks x,y,z = {landmark_dim} dims, matching extract_landmarks.py."
        )

    t = arr.shape[0]
    if t < max_frames:
        pad = np.zeros((max_frames - t, landmark_dim), dtype=np.float32)
        mask = np.concatenate([np.ones(t), np.zeros(max_frames - t)]).astype(np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:max_frames]
        mask = np.ones(max_frames, dtype=np.float32)

    return arr, mask


@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    await websocket.accept()
    model = _state["model"]
    device = _state["device"]
    max_frames = _state["max_frames"]
    landmark_dim = _state["landmark_dim"]
    confidence_threshold = _state["confidence_threshold"]

    try:
        while True:
            payload = await websocket.receive_json()
            raw_landmarks = payload.get("landmarks")

            if not raw_landmarks:
                await websocket.send_json({"error": "missing 'landmarks' field"})
                continue

            try:
                arr, mask = preprocess_landmarks(raw_landmarks, max_frames, landmark_dim)
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
                continue

            landmarks_t = torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, T, 345)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)       # (1, T)

            with torch.no_grad():
                raw_translation = model.generate(landmarks_t, mask_t, num_beams=4)[0]

            corrected, confidence = correct_grammar(raw_translation)

            if confidence < confidence_threshold:
                await websocket.send_json({
                    "sentence": None,
                    "intent": None,
                    "raw_translation": raw_translation,
                    "confidence": confidence,
                    "message": "Low confidence translation - please re-sign.",
                })
                continue

            intent = classify_intent(corrected)

            await websocket.send_json({
                "sentence": corrected,
                "intent": intent,
                "raw_translation": raw_translation,
                "confidence": confidence,
            })

    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}
