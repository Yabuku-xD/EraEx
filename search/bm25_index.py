import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


class BM25Index:
    def __init__(self):
        self.index = None
        self.doc_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self._loaded = False
        self._backend = None
    
    def build(self, documents: List[Dict], text_field: str = "doc_text_music", id_field: str = "track_id"):
        self.doc_ids = []
        self.corpus_tokens = []
        
        for doc in documents:
            doc_id = doc.get(id_field, "")
            text = doc.get(text_field, "") or ""
            tokens = text.lower().split()
            
            self.doc_ids.append(str(doc_id))
            self.corpus_tokens.append(tokens)
        
        if BM25S_AVAILABLE:
            self._backend = "bm25s"
            corpus_tokens_array = bm25s.tokenize([" ".join(t) for t in self.corpus_tokens])
            self.index = bm25s.BM25()
            self.index.index(corpus_tokens_array)
        elif RANK_BM25_AVAILABLE:
            self._backend = "rank_bm25"
            self.index = BM25Okapi(self.corpus_tokens)
        else:
            raise ImportError("Neither bm25s nor rank_bm25 is installed. Run: pip install bm25s")
        
        self._loaded = True
        print(f"BM25 index built with {len(self.doc_ids)} documents using {self._backend}")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict]:
        if not self._loaded or self.index is None:
            return []
        
        query_tokens = query.lower().split()
        
        if self._backend == "bm25s":
            query_tokens_array = bm25s.tokenize([query])
            results, scores = self.index.retrieve(query_tokens_array, k=min(top_k, len(self.doc_ids)))
            
            output = []
            for i in range(len(results[0])):
                idx = results[0][i]
                if idx < len(self.doc_ids):
                    output.append({
                        "track_id": self.doc_ids[idx],
                        "bm25_score": float(scores[0][i]),
                        "index": idx
                    })
            return output
        
        elif self._backend == "rank_bm25":
            scores = self.index.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            output = []
            for idx in top_indices:
                if scores[idx] > 0:
                    output.append({
                        "track_id": self.doc_ids[idx],
                        "bm25_score": float(scores[idx]),
                        "index": idx
                    })
            return output
        
        return []
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "doc_ids": self.doc_ids,
            "corpus_tokens": self.corpus_tokens,
            "backend": self._backend
        }
        
        if self._backend == "bm25s" and self.index is not None:
            index_dir = path.parent / f"{path.stem}_bm25s"
            self.index.save(str(index_dir))
            save_data["bm25s_path"] = str(index_dir)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"BM25 index saved to {path}")
    
    def load(self, path: Path):
        path = Path(path)
        if not path.exists():
            print(f"BM25 index not found at {path}")
            return False
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.doc_ids = save_data["doc_ids"]
        self.corpus_tokens = save_data["corpus_tokens"]
        self._backend = save_data.get("backend", "rank_bm25")
        
        if self._backend == "bm25s" and BM25S_AVAILABLE:
            bm25s_path = save_data.get("bm25s_path")
            if bm25s_path and Path(bm25s_path).exists():
                self.index = bm25s.BM25.load(bm25s_path, load_corpus=False)
            else:
                corpus_tokens_array = bm25s.tokenize([" ".join(t) for t in self.corpus_tokens])
                self.index = bm25s.BM25()
                self.index.index(corpus_tokens_array)
        elif RANK_BM25_AVAILABLE:
            self._backend = "rank_bm25"
            self.index = BM25Okapi(self.corpus_tokens)
        else:
            raise ImportError("No BM25 backend available")
        
        self._loaded = True
        print(f"BM25 index loaded: {len(self.doc_ids)} documents ({self._backend})")
        return True


bm25_index = BM25Index()
