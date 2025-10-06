# API_model.py
"""
Embedding + similarity via Hugging Face Inference API (no local models).
Expose: calculate_similarity_api(text_a, text_b) -> float (0..100)
Env:
  EMBED_MODEL  : default "sentence-transformers/all-MiniLM-L6-v2"
  HF_TOKEN     : required (set in Space/host env)
  MAX_TEXT_LEN : default "20000"
  REQ_TIMEOUT  : default "40"
"""

import os
import re
import numpy as np
from huggingface_hub import InferenceClient

# -------- App config --------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN    = os.getenv("HF_TOKEN")                      # set in host env / Spaces Variables
MAX_LEN     = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

# HF Inference API client (timeout belongs on the client)
_hf = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)

def _trim(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[:MAX_LEN]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed_api(text: str) -> np.ndarray:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (add it in environment or Space → Settings → Variables).")
    feats = _hf.feature_extraction(_trim(text))  # API returns list/array
    arr = np.array(feats, dtype=np.float32)
    # If token-level embeddings returned, mean-pool to sentence vector
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape from the Inference API: {arr.shape}")
    return arr

def calculate_similarity_api(text_a: str, text_b: str) -> float:
    """
    Return cosine similarity (%) between two texts using HF Inference API embeddings.
    """
    a_vec = _embed_api(text_a)
    b_vec = _embed_api(text_b)
    score = _cosine(a_vec, b_vec) * 100.0
    return float(np.round(score, 2))
