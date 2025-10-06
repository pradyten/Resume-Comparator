# API_model.py
import os
import re
import numpy as np
from typing import Optional
from huggingface_hub import InferenceClient

# -------- Config (via env) --------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN    = os.getenv("HF_TOKEN")                      # set in Space/CI env
MAX_LEN     = int(os.getenv("MAX_TEXT_LEN", "20000"))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", "40"))

# Lazy client (created on first use to avoid import-time failures)
_hf_client: Optional[InferenceClient] = None

def _get_client() -> InferenceClient:
    global _hf_client
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (add it in environment/Secrets).")
    if _hf_client is None:
        _hf_client = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=REQ_TIMEOUT)
    return _hf_client

# -------- Utilities --------
def _trim(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[:MAX_LEN]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed_api(text: str) -> np.ndarray:
    """Return a 1D embedding using HF Inference API (mean-pool if token-level)."""
    client = _get_client()
    feats = client.feature_extraction(_trim(text))
    arr = np.array(feats, dtype=np.float32)
    if arr.ndim == 2:                       # token-level → mean pool
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise RuntimeError("Unexpected embedding shape from the Inference API.")
    return arr

# -------- Public API (drop-in for local similarity) --------
def calculate_similarity_api(resume_text: str, job_text: str) -> float:
    """
    Returns similarity in % (0-100), matching the signature/scale used in local_model.
    Assumes input strings are already preprocessed upstream (lowercased, stopwords removed, etc.).
    """
    r_vec = _embed_api(resume_text)
    j_vec = _embed_api(job_text)
    score = _cosine(r_vec, j_vec) * 100.0
    return float(np.round(score, 2))

def api_healthcheck() -> str:
    """Optional: ping once to verify credentials/model availability."""
    try:
        _ = _embed_api("healthcheck")
        return f"OK: Using {EMBED_MODEL}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
