import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM


def load_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def embed_texts(
    model: SentenceTransformer,
    texts: list,
    batch_size: int = 256,
    show_progress: bool = True
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype(np.float16)


def save_embeddings(
    embeddings: np.ndarray,
    output_path: Path,
    use_float16: bool = True
):
    if use_float16:
        embeddings = embeddings.astype(np.float16)
    np.save(output_path, embeddings)
    print(f"Saved embeddings: {output_path} ({embeddings.shape})")


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)
