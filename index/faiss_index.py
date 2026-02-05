import faiss
import numpy as np
from pathlib import Path

from config.settings import EMBEDDING_DIM


def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def build_ivf_pq_index(
    embeddings: np.ndarray,
    n_list: int = 100,
    m_subquantizers: int = 48,
    n_bits: int = 8,
    n_train_samples: int = None
) -> faiss.Index:
    embeddings = embeddings.astype(np.float32)
    n_vectors = embeddings.shape[0]
    
    if n_train_samples is None:
        n_train_samples = min(n_vectors, max(n_list * 40, 100000))
    
    if n_vectors < n_list * 40:
        n_list = max(1, n_vectors // 40)
    
    if EMBEDDING_DIM % m_subquantizers != 0:
        m_subquantizers = 32
    
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, n_list, m_subquantizers, n_bits)
    
    if n_train_samples < n_vectors:
        train_indices = np.random.choice(n_vectors, n_train_samples, replace=False)
        train_data = embeddings[train_indices]
    else:
        train_data = embeddings
    
    print(f"  Training IVF+PQ: n_list={n_list}, m={m_subquantizers}, training on {len(train_data):,} samples")
    index.train(train_data)
    
    print(f"  Adding {n_vectors:,} vectors...")
    index.add(embeddings)
    
    return index


def save_index(index: faiss.Index, path: Path):
    faiss.write_index(index, str(path))
    print(f"  Saved index: {path}")


def load_index(path: Path, mmap: bool = True) -> faiss.Index:
    if mmap:
        return faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
    return faiss.read_index(str(path))


def search_index(index: faiss.Index, query_embedding: np.ndarray, k: int = 100, n_probe: int = 10) -> tuple:
    query = query_embedding.astype(np.float32).reshape(1, -1)
    
    if hasattr(index, 'nprobe'):
        index.nprobe = n_probe
    
    scores, indices = index.search(query, k)
    return scores[0], indices[0]