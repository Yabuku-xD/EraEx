import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    _instance = None
    _model = None

    # Create or reuse the singleton instance.
    def __new__(cls):
        """
        Create and return a new instance.
        
        This method implements the new step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Load model.
    def load_model(self):
        """
        Load model.
        
        This method implements the load model step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if settings.MOCK_MODE:
            return None
        if self._model is None:
            model_name = settings.EMBEDDING_MODEL
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self._model = SentenceTransformer(model_name, device=device)
            except Exception as exc:
                msg = str(exc)
                if (
                    "serious vulnerability issue in `torch.load`" in msg
                    and "upgrade torch to at least v2.6" in msg
                ):
                    raise RuntimeError(
                        "Embedding model load failed because installed torch is too old for "
                        "current transformers security checks. Upgrade torch to >=2.6, then "
                        "retry. Example (CPU): `python -m pip install --upgrade \"torch>=2.6\"`."
                    ) from exc
                logger.error("Embedding model load failed: %s", exc)
                raise
        return self._model

    # Encode this operation.
    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        """
        Execute encode.
        
        This method implements the encode step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if settings.MOCK_MODE:
            n = len(texts)
            dim = settings.EMBEDDING_DIM
            vecs = np.random.rand(n, dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs /= norm
            return vecs
        model = self.load_model()
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )


embedding_handler = EmbeddingHandler()