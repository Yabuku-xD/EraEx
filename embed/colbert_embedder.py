import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_DIM = 128
MAX_DOC_TOKENS = 180
MAX_QUERY_TOKENS = 32


class ColBERTEmbedder:
    def __init__(self, model_name: str = COLBERT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
        
        print(f"Loading ColBERT model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        print("ColBERT model loaded!")
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        if not self._loaded:
            self.load()
        
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding documents")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_DOC_TOKENS,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                token_embeddings = outputs.last_hidden_state
                
                attention_mask = inputs["attention_mask"]
                
                for j in range(len(batch_texts)):
                    mask = attention_mask[j].bool()
                    doc_tokens = token_embeddings[j][mask]
                    
                    doc_tokens = doc_tokens / (doc_tokens.norm(dim=1, keepdim=True) + 1e-9)
                    
                    all_embeddings.append(doc_tokens.cpu().numpy().astype(np.float16))
        
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        if not self._loaded:
            self.load()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=MAX_QUERY_TOKENS,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            token_embeddings = outputs.last_hidden_state[0]
            mask = inputs["attention_mask"][0].bool()
            query_tokens = token_embeddings[mask]
            
            query_tokens = query_tokens / (query_tokens.norm(dim=1, keepdim=True) + 1e-9)
            
            return query_tokens.cpu().numpy().astype(np.float32)
    
    def embed_query_pooled(self, query: str) -> np.ndarray:
        if not self._loaded:
            self.load()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=MAX_QUERY_TOKENS,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            token_embeddings = outputs.last_hidden_state[0]
            mask = inputs["attention_mask"][0].bool()
            valid_tokens = token_embeddings[mask]
            
            pooled = valid_tokens.mean(dim=0)
            pooled = pooled / (pooled.norm() + 1e-9)
            
            return pooled.cpu().numpy().astype(np.float32)
    
    def maxsim_score(
        self,
        query_tokens: np.ndarray,
        doc_tokens: np.ndarray
    ) -> float:
        similarities = query_tokens @ doc_tokens.T
        
        max_sims = similarities.max(axis=1)
        
        return float(max_sims.sum())


class SimplifiedColBERTEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._loaded = True
        print("Model loaded!")
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = True
    ) -> np.ndarray:
        if not self._loaded:
            self.load()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings.astype(np.float16)
    
    def embed_query(self, query: str) -> np.ndarray:
        if not self._loaded:
            self.load()
        
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.astype(np.float32)


def get_embedder(use_colbert: bool = True):
    if use_colbert and TRANSFORMERS_AVAILABLE:
        try:
            return ColBERTEmbedder()
        except Exception as e:
            print(f"ColBERT initialization failed: {e}")
            print("Falling back to simplified embedder")
    
    return SimplifiedColBERTEmbedder()