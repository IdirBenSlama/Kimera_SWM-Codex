from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.core.primitives.constants import EMBEDDING_DIM

try:
    import onnxruntime as ort
except Exception:
    ort = None  # type: ignore

try:
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    F = None  # type: ignore

# FlagEmbedding is optional, handle separately
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:
    BGEM3FlagModel = None  # type: ignore

# --- Setup Logging ---
log = logging.getLogger(__name__)
class _DummyTransformer:
    """Auto-generated class."""
    pass
    def encode(self, text: str, max_length: int = 512):
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(float)
        reps = (EMBEDDING_DIM + len(vec) - 1) // len(vec)
        return np.tile(vec, reps)[:EMBEDDING_DIM] / 255.0


# --- Globals ---
_embedding_model = None
_model_lock = Lock()

# --- Performance Tracking ---
_performance_stats = {
    "total_embeddings": 0
    "total_time": 0.0
    "avg_time_per_embedding": 0.0
    "model_load_time": 0.0
}
_stats_lock = Lock()

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Embedding model will use device: {DEVICE.upper()}")

# --- Environment Configuration ---
MODEL_NAME = "BAAI/bge-m3"
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "./models/bge-m3-onnx")
LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_EMBEDDING", "0") == "1"
USE_ONNX = os.getenv("USE_ONNX", "1") == "1"
USE_FLAG_EMBEDDING = os.getenv("USE_FLAG_EMBEDDING", "1") == "1"
MAX_LENGTH = int(os.getenv("MAX_EMBEDDING_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


def _lightweight_encoder(text: str) -> List[float]:
    """Create a consistent random vector based on the hash of the text."""
    seed = hash(text) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    return rng.rand(EMBEDDING_DIM).tolist()


def get_performance_stats() -> Dict[str, float]:
    """Get embedding performance statistics (thread-safe)."""
    with _stats_lock:
        return _performance_stats.copy()
class EmbeddingModelWrapper:
    """Auto-generated class."""
    pass
    """Wrapper to provide a consistent .encode() interface."""

    def __init__(self, model):
        self._model = model

    def encode(self, text: str):
        """Encode text using the underlying model."""
        embedding_list = encode_text(text)
        # Return as numpy array so .tolist() works in the API
        return np.array(embedding_list)


def get_embedding_model():
    """Returns the embedding model, initializing it if necessary."""
    model = _get_model()
    return EmbeddingModelWrapper(model)


def initialize_embedding_model():
    """Explicitly initializes the embedding model with optimized loading strategy."""
    global _embedding_model, _performance_stats
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                start_time = time.time()

                # Priority 1: Try FlagEmbedding BGE-M3 (most optimized)
                if USE_FLAG_EMBEDDING and BGEM3FlagModel is not None:
                    log.info(
                        f"Initializing FlagEmbedding BGE-M3 model on {DEVICE.upper()}..."
                    )
                    try:
                        _embedding_model = {
                            "flag_model": BGEM3FlagModel(
                                MODEL_NAME, use_fp16=DEVICE == "cuda"
                            ),
                            "type": "flag_embedding",
                        }
                        log.info("FlagEmbedding BGE-M3 model loaded successfully.")
                    except Exception as e:
                        log.warning(
                            f"Failed to load FlagEmbedding model: {e}. Falling back to ONNX."
                        )
                        _embedding_model = None

                # Priority 2: Try ONNX Runtime (optimized inference)
                if _embedding_model is None and USE_ONNX and ort is not None:
                    log.info(
                        f"Initializing ONNX embedding model '{MODEL_NAME}' on {DEVICE.upper()}..."
                    )
                    try:
                        onnx_path = Path(ONNX_MODEL_PATH) / "model.onnx"
                        if onnx_path.exists():
                            providers = (
                                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                                if DEVICE == "cuda"
                                else ["CPUExecutionProvider"]
                            )
                            _embedding_model = {
                                "session": ort.InferenceSession(
                                    str(onnx_path), providers=providers
                                ),
                                "tokenizer": AutoTokenizer.from_pretrained(MODEL_NAME),
                                "type": "onnx",
                            }
                            log.info("ONNX embedding model loaded successfully.")
                        else:
                            log.warning(
                                f"ONNX model not found at {onnx_path}. Falling back to Transformers."
                            )
                            _embedding_model = None
                    except Exception as e:
                        log.warning(
                            f"Failed to load ONNX model: {e}. Falling back to Transformers model."
                        )
                        _embedding_model = None

                # Priority 3: Standard Transformers model
                if _embedding_model is None:
                    log.info(
                        f"Initializing Transformers embedding model '{MODEL_NAME}' on {DEVICE.upper()}..."
                    )
                    try:
                        model = AutoModel.from_pretrained(
                            MODEL_NAME
                            torch_dtype=(
                                torch.float16 if DEVICE == "cuda" else torch.float32
                            ),
                        )
                        _embedding_model = {
                            "model": model.to(DEVICE),
                            "tokenizer": AutoTokenizer.from_pretrained(MODEL_NAME),
                            "type": "transformers",
                        }
                        log.info("Transformers embedding model loaded successfully.")
                    except Exception as e:
                        log.error(
                            f"Failed to load any embedding model: {e}. Using dummy encoder."
                        )
                        _embedding_model = {"type": "dummy"}

                load_time = time.time() - start_time
                with _stats_lock:
                    _performance_stats["model_load_time"] = load_time
                log.info(f"Model initialization completed in {load_time:.2f}s")

    return _embedding_model


def _get_model():
    """Initializes and returns the embedding model (thread-safe)."""
    global _embedding_model
    if _embedding_model is None:
        return initialize_embedding_model()
    return _embedding_model


def encode_text(
    text: str, device: Optional[str] = None
) -> Union[np.ndarray, List[float]]:
    """Encodes a string of text into a vector embedding with performance tracking."""
    global _performance_stats

    # Use the provided device or fall back to the global default
    target_device = device if device is not None else DEVICE

    if LIGHTWEIGHT_MODE:
        return _lightweight_encoder(text)

    start_time = time.time()
    model_type = "unknown"
    embedding = []

    try:
        model = _get_model()
        model_type = (
            model.get("type", "unknown") if isinstance(model, dict) else "unknown"
        )

        # Priority 1: FlagEmbedding BGE-M3 (most optimized)
        if isinstance(model, dict) and model.get("type") == "flag_embedding":
            try:
                flag_model = model["flag_model"]
                # FlagEmbedding handles device internally based on its initialization
                embedding_array = flag_model.encode([text])["dense_vecs"][0]
                # Ensure it's a numpy array on CPU
                if hasattr(embedding_array, "cpu"):
                    return embedding_array.cpu().numpy()
                elif hasattr(embedding_array, "numpy"):
                    return embedding_array.numpy()
                else:
                    return np.array(embedding_array)
            except Exception as e:
                log.error(
                    f"FlagEmbedding inference failed: {e}. Falling back to next method."
                )

        # Priority 2: ONNX Runtime inference
        elif isinstance(model, dict) and model.get("type") == "onnx":
            try:
                tokenizer = model["tokenizer"]
                session = model["session"]
                inputs = tokenizer(
                    text
                    return_tensors="np",
                    padding=True
                    truncation=True
                    max_length=MAX_LENGTH
                )
                outputs = session.run(
                    None
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                )
                embedding_array = outputs[0][0]
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm
                embedding = embedding_array.tolist()
                return embedding
            except Exception as e:
                log.error(f"ONNX inference failed: {e}. Falling back to Transformers.")

        # Priority 3: Transformers model inference
        elif isinstance(model, dict) and model.get("type") == "transformers":
            try:
                tokenizer = model["tokenizer"]
                transformer_model = model["model"].to(
                    target_device
                )  # Ensure model is on correct device
                inputs = tokenizer(
                    text
                    return_tensors="pt",
                    padding=True
                    truncation=True
                    max_length=MAX_LENGTH
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = transformer_model(**inputs)
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    )
                    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embedding_tensor = sum_embeddings / sum_mask
                    embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)
                # Return as numpy array (always on CPU for compatibility)
                return embedding_tensor[0].cpu().numpy()
            except Exception as e:
                log.error(
                    f"Transformers inference failed: {e}. Falling back to dummy encoder."
                )

        # Fallback to dummy encoder
        log.warning("No valid embedding model available. Using dummy encoder.")
        model_type = "dummy"
        embedding = _DummyTransformer().encode(text)
        return embedding

    finally:
        # Update performance statistics
        inference_time = time.time() - start_time
        with _stats_lock:
            _performance_stats["total_embeddings"] += 1
            _performance_stats["total_time"] += inference_time
            if _performance_stats["total_embeddings"] > 0:
                _performance_stats["avg_time_per_embedding"] = (
                    _performance_stats["total_time"]
                    / _performance_stats["total_embeddings"]
                )


def encode_batch(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of texts into a list of vector embeddings.
    This function now uses a cached helper to handle the non-hashable list argument.
    """
    # lru_cache requires hashable arguments, so we convert the list to a tuple.
    return _encode_batch_cached(tuple(texts))


@functools.lru_cache(maxsize=128)
def _encode_batch_cached(texts: tuple[str]) -> List[List[float]]:
    """Cached implementation for batch encoding."""
    global _performance_stats

    texts_list = list(texts)  # Convert back to list for processing

    if LIGHTWEIGHT_MODE:
        return [_lightweight_encoder(text) for text in texts_list]

    start_time = time.time()
    model_type = "unknown"

    try:
        model = _get_model()
        model_type = (
            model.get("type", "unknown") if isinstance(model, dict) else "unknown"
        )

        # Batch processing for FlagEmbedding
        if isinstance(model, dict) and model.get("type") == "flag_embedding":
            try:
                flag_model = model["flag_model"]
                embeddings = flag_model.encode(texts_list)["dense_vecs"]
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                log.error(
                    f"FlagEmbedding batch inference failed: {e}. Falling back to individual processing."
                )

        # Fallback to individual processing if batch is not supported or fails
        return [encode_text(text) for text in texts_list]

    finally:
        # Update performance statistics for the batch
        inference_time = time.time() - start_time
        with _stats_lock:
            _performance_stats["total_embeddings"] += len(texts_list)
            _performance_stats["total_time"] += inference_time
            if _performance_stats["total_embeddings"] > 0:
                _performance_stats["avg_time_per_embedding"] = (
                    _performance_stats["total_time"]
                    / _performance_stats["total_embeddings"]
                )


def extract_semantic_features(text: str) -> Dict[str, float]:
    """
    Extracts semantic features from a text string.
    Handles both CPU and GPU tensors by converting them properly to numpy arrays.
    """
    try:
        embedding_raw = encode_text(text)
        log.info(
            f"Raw embedding type: {type(embedding_raw)}, has cpu: {hasattr(embedding_raw, 'cpu')}, has numpy: {hasattr(embedding_raw, 'numpy')}"
        )

        # Handle different return types from encode_text
        if hasattr(embedding_raw, "cpu"):  # PyTorch tensor on GPU
            log.info("Converting GPU tensor to CPU numpy array")
            embedding = embedding_raw.cpu().numpy()
        elif hasattr(embedding_raw, "numpy"):  # PyTorch tensor on CPU
            log.info("Converting CPU tensor to numpy array")
            embedding = embedding_raw.numpy()
        elif isinstance(embedding_raw, np.ndarray):  # Already numpy array
            log.info("Already numpy array")
            embedding = embedding_raw
        elif isinstance(embedding_raw, list):  # List of floats
            log.info("Converting list to numpy array")
            embedding = np.array(embedding_raw)
        else:
            # Fallback: try to convert to numpy array
            log.warning(f"Unknown type {type(embedding_raw)}, attempting conversion")
            try:
                embedding = np.array(embedding_raw)
            except Exception as e:
                log.error(f"Failed to convert embedding to numpy array: {e}")
                # Return default features
                return {
                    "mean": 0.0
                    "std": 0.0
                    "norm": 0.0
                    "error": f"Conversion failed: {str(e)}",
                }
    except Exception as e:
        log.error(f"Error in encode_text call: {e}")
        return {
            "mean": 0.0
            "std": 0.0
            "norm": 0.0
            "error": f"Encode_text failed: {str(e)}",
        }

    # Calculate semantic features
    try:
        features = {
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding)),
            "norm": float(np.linalg.norm(embedding)),
            "max": float(np.max(embedding)),
            "min": float(np.min(embedding)),
            "dimensionality": len(embedding) if hasattr(embedding, "__len__") else 0
        }
        return features
    except Exception as e:
        log.error(f"Error calculating semantic features: {e}")
        return {
            "mean": 0.0
            "std": 0.0
            "norm": 0.0
            "error": f"Feature calculation failed: {str(e)}",
        }
