import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
from src.config.settings import SBERT_MODEL_NAME


class Embedder:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Embedder._initialized:
            return
        self._model: Optional[SentenceTransformer] = None
        Embedder._initialized = True

    def _ensure_model(self):
        if self._model is None:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self._model = SentenceTransformer(SBERT_MODEL_NAME)
            print("SBERT model loaded.")

    def encode(self, text: str) -> np.ndarray:
        self._ensure_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype("float32")

    def encode_batch(self, texts: list) -> np.ndarray:
        self._ensure_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")

    def is_ready(self) -> bool:
        return self._model is not None


embedder = Embedder()