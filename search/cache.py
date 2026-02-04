import redis
import numpy as np
import json
from typing import Optional
import hashlib


class EmbeddingCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 86400,  # 24 hours
        prefix: str = "eraex:embed:"
    ):
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            print(f"Redis cache connected ({host}:{port})")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Redis unavailable, using in-memory fallback: {e}")
            self.redis_client = None
    
    def _make_key(self, query: str) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.prefix}{query_hash}"
    
    def get(self, query: str) -> Optional[np.ndarray]:
        key = self._make_key(query)
        
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    embedding = np.frombuffer(data, dtype=np.float32)
                    return embedding
            except redis.RedisError:
                pass
        else:
            # Fallback to in-memory
            if key in self.fallback_cache:
                return self.fallback_cache[key]
        
        return None
    
    def set(self, query: str, embedding: np.ndarray) -> bool:
        key = self._make_key(query)
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, embedding_bytes)
                return True
            except redis.RedisError:
                pass
        
        # Fallback to in-memory (limit size to prevent memory issues)
        if len(self.fallback_cache) < 10000:
            self.fallback_cache[key] = embedding
        return True
    
    def clear(self) -> int:
        count = 0
        if self.redis_client:
            try:
                keys = self.redis_client.keys(f"{self.prefix}*")
                if keys:
                    count = self.redis_client.delete(*keys)
            except redis.RedisError:
                pass
        else:
            count = len(self.fallback_cache)
            self.fallback_cache.clear()
        return count
    
    def stats(self) -> dict:
        if self.redis_client:
            try:
                keys = self.redis_client.keys(f"{self.prefix}*")
                return {
                    "backend": "redis",
                    "cached_queries": len(keys),
                    "ttl_seconds": self.ttl
                }
            except redis.RedisError:
                pass
        
        return {
            "backend": "in-memory",
            "cached_queries": len(self.fallback_cache),
            "max_size": 10000
        }

# Global cache instance
embedding_cache = EmbeddingCache()