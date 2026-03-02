import json
import logging
import os
import re
import socket
import time
import urllib.parse
import urllib.request


class LyricsProviderEngine:
    # Initialize lyrics provider settings and runtime caches/state.
    def __init__(self):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._cache = {}
        ttl_env = (
            os.getenv("LYRICS_CACHE_TTL_SEC")
            if os.getenv("LYRICS_CACHE_TTL_SEC") is not None
            else os.getenv("CACHE_TTL")
        )
        self.cache_ttl_sec = max(
            30, int(float(ttl_env if ttl_env is not None else "86400") or 86400)
        )
        self.cache_max = max(16, int(float(os.getenv("LYRICS_CACHE_MAX", "5000") or 5000)))
        cache_enabled_env = os.getenv("LYRICS_CACHE_ENABLED")
        if cache_enabled_env is None:
            self.cache_enabled = ttl_env is not None
        else:
            self.cache_enabled = (
                str(cache_enabled_env or "").strip().lower() in {"1", "true", "yes", "y", "on"}
            )
        self.lookup_budget_sec = max(
            2.0,
            float(os.getenv("LYRICS_LOOKUP_BUDGET_SEC", "20.0") or 20.0),
        )
        self.lrclib_get_timeout_sec = max(
            1.0,
            float(os.getenv("LRCLIB_GET_TIMEOUT_SEC", "3.0") or 3.0),
        )
        self.lrclib_search_timeout_sec = max(
            1.0,
            float(os.getenv("LRCLIB_SEARCH_TIMEOUT_SEC", "12.0") or 12.0),
        )
        self._timeout_streak = 0
        self._backoff_until = 0.0
        self.provider_timeout_backoff_sec = max(
            5,
            int(float(os.getenv("LYRICS_PROVIDER_TIMEOUT_BACKOFF_SEC", "20") or 20)),
        )

    # Report current engine status for diagnostics.
    def health(self):
        """
        Execute health.
        
        This method implements the health step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return {
            "lyrics_provider_engine": "lrclib_direct",
            "lyrics_cache_enabled": bool(self.cache_enabled),
            "lyrics_cache_ttl_sec": int(self.cache_ttl_sec),
            "lyrics_lookup_budget_sec": float(self.lookup_budget_sec),
            "lyrics_provider_backoff_active": bool(time.time() < self._backoff_until),
        }

    # Fetch lyrics payload with optional fast/plain-only modes for UI usage.
    def lookup(self, title="", artist="", fast_mode=False, plain_only=False):
        """
        Execute lookup.
        
        This method implements the lookup step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        title = str(title or "").strip()
        artist = str(artist or "").strip()
        fast_mode = bool(fast_mode)
        plain_only = bool(plain_only)
        if not title and not artist:
            return {"synced": False, "lyrics": "", "provider": ""}

        cached = self._get_cached(title, artist)
        if cached is not None:
            if plain_only and bool(cached.get("synced")):
                plain_cached = self._strip_lrc_timestamps(cached.get("lyrics") or "")
                if plain_cached:
                    return {
                        "synced": False,
                        "lyrics": plain_cached,
                        "provider": str(cached.get("provider") or "lrclib:cache")
                        + ":plain",
                    }
            return cached

        deadline = time.time() + self.lookup_budget_sec

        # Exact lookup is helpful for synced lyrics, but can be slower and less useful in fast/plain mode.
        if not fast_mode:
            exact_titles = []
            seen_exact_titles = set()
            for candidate in [title, self._norm_text(title)]:
                q_title = str(candidate or "").strip()
                q_key = q_title.lower()
                if not q_title or q_key in seen_exact_titles:
                    continue
                seen_exact_titles.add(q_key)
                exact_titles.append(q_title)
            for q_title in exact_titles:
                if time.time() >= deadline:
                    break
                try:
                    exact = self._lrclib_get(track_name=q_title, artist_name=artist)
                except Exception:
                    exact = None
                if isinstance(exact, dict) and (
                    exact.get("syncedLyrics") or exact.get("plainLyrics")
                ):
                    payload = self._payload_from_item(
                        exact,
                        provider="lrclib:get",
                        plain_only=plain_only,
                    )
                    if payload.get("lyrics"):
                        self._set_cached(title, artist, payload)
                        return payload

        norm_title = self._norm_text(title)
        queries = []
        seen_queries = set()
        for q_title, q_artist in [
            (title, artist),
            (norm_title, artist),
            (title, ""),
            (norm_title, ""),
        ]:
            key = (str(q_title or "").strip(), str(q_artist or "").strip())
            if key in seen_queries:
                continue
            seen_queries.add(key)
            if not key[0] and not key[1]:
                continue
            queries.append(key)

        best_match = None
        best_score = -1
        for q_title, q_artist in queries:
            if time.time() >= deadline:
                break
            try:
                results = self._lrclib_search(track_name=q_title, artist_name=q_artist)
            except Exception:
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                if not (item.get("syncedLyrics") or item.get("plainLyrics")):
                    continue
                score = self._score_match(item, title, artist)
                if plain_only:
                    if item.get("plainLyrics"):
                        score += 3
                    if item.get("syncedLyrics") and not item.get("plainLyrics"):
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_match = item
            if best_match and best_score >= 8:
                break

        payload = (
            self._payload_from_item(
                best_match,
                provider="lrclib:search",
                plain_only=plain_only,
            )
            if best_match
            else {"synced": False, "lyrics": "", "provider": ""}
        )
        # Cache only successful lyric fetches so timeout-driven empties do not poison retries.
        if payload.get("lyrics"):
            self._set_cached(title, artist, payload)
        return payload

    # Remove a cached lyrics entry for a title/artist (used after actual play starts).
    def invalidate(self, title="", artist=""):
        """
        Execute invalidate.
        
        This method implements the invalidate step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        key = self._cache_key(title, artist)
        self._cache.pop(key, None)

    # Check whether an exception looks like a provider timeout.
    @staticmethod
    def is_timeout_exception(exc):
        """
        Return whether timeout exception.
        
        This method implements the is timeout exception step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(exc, socket.timeout):
            return True
        text = str(exc or "").lower()
        return "timed out" in text or "timeout" in text

    # Normalize track/artist text for matching and cache keys.
    def _norm_text(self, value):
        """
        Execute norm text.
        
        This method implements the norm text step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"\s*[\(\[\-–—].*?(official|lyrics?|audio|video).*?[\)\]]?\s*$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\s*[\(\[]?\s*(feat\.?|featuring|ft\.)\s+[^)\]]+[\)\]]?\s*$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"[^a-z0-9 ]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Build a stable cache key from normalized title and artist.
    def _cache_key(self, title, artist):
        """
        Execute cache key.
        
        This method implements the cache key step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return (self._norm_text(title), self._norm_text(artist))

    # Read successful cached lyrics when enabled and not expired.
    def _get_cached(self, title, artist):
        """
        Get cached.
        
        This method implements the get cached step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self.cache_enabled:
            return None
        key = self._cache_key(title, artist)
        hit = self._cache.get(key)
        if not hit:
            return None
        if (time.time() - float(hit.get("ts", 0))) > self.cache_ttl_sec:
            self._cache.pop(key, None)
            return None
        return hit.get("payload")

    # Save successful lyrics to cache when enabled.
    def _set_cached(self, title, artist, payload):
        """
        Set cached.
        
        This method implements the set cached step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self.cache_enabled:
            return
        key = self._cache_key(title, artist)
        if len(self._cache) >= self.cache_max:
            try:
                oldest_key = next(iter(self._cache.keys()))
                self._cache.pop(oldest_key, None)
            except Exception:
                self._cache.clear()
        self._cache[key] = {"ts": time.time(), "payload": payload}

    # Return whether provider requests should be skipped due to timeout backoff.
    def _provider_available(self):
        """
        Execute provider available.
        
        This method implements the provider available step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return time.time() >= self._backoff_until

    # Increase timeout streak and activate temporary provider backoff.
    def _record_timeout(self):
        """
        Execute record timeout.
        
        This method implements the record timeout step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._timeout_streak += 1
        if self._timeout_streak >= 3:
            self._backoff_until = time.time() + self.provider_timeout_backoff_sec

    # Reset timeout streak after a successful provider response.
    def _record_success(self):
        """
        Execute record success.
        
        This method implements the record success step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._timeout_streak = 0
        self._backoff_until = 0.0

    # Perform an LRCLIB exact lookup request.
    def _lrclib_get(self, track_name="", artist_name=""):
        """
        Execute lrclib get.
        
        This method implements the lrclib get step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        q_track = str(track_name or "").strip()
        q_artist = str(artist_name or "").strip()
        if not q_track:
            return None
        if not self._provider_available():
            return None
        url = (
            "https://lrclib.net/api/get?"
            f"track_name={urllib.parse.quote(q_track)}&artist_name={urllib.parse.quote(q_artist)}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "EraEx Music App",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.lrclib_get_timeout_sec) as response:
                data = json.loads(response.read().decode())
            self._record_success()
        except Exception as exc:
            if self.is_timeout_exception(exc):
                self._record_timeout()
            raise
        return data if isinstance(data, dict) else None

    # Perform an LRCLIB search request when exact lookup misses.
    def _lrclib_search(self, track_name="", artist_name=""):
        """
        Execute lrclib search.
        
        This method implements the lrclib search step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        q_track = str(track_name or "").strip()
        q_artist = str(artist_name or "").strip()
        if not q_track and not q_artist:
            return []
        if not self._provider_available():
            return []
        url = (
            "https://lrclib.net/api/search?"
            f"track_name={urllib.parse.quote(q_track)}&artist_name={urllib.parse.quote(q_artist)}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "EraEx Music App",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.lrclib_search_timeout_sec
            ) as response:
                data = json.loads(response.read().decode())
            self._record_success()
        except Exception as exc:
            if self.is_timeout_exception(exc):
                self._record_timeout()
                # One rescue retry with a slightly larger timeout helps on slow LRCLIB responses.
                retry_timeout = max(self.lrclib_search_timeout_sec + 3.0, 10.0)
                try:
                    with urllib.request.urlopen(req, timeout=retry_timeout) as response:
                        data = json.loads(response.read().decode())
                    self._record_success()
                    return data if isinstance(data, list) else []
                except Exception as retry_exc:
                    if self.is_timeout_exception(retry_exc):
                        self._record_timeout()
                    raise retry_exc
            raise
        return data if isinstance(data, list) else []

    # Build API payload from an LRCLIB result with optional plain-only conversion.
    def _payload_from_item(self, item, provider, plain_only=False):
        """
        Execute payload from item.
        
        This method implements the payload from item step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not isinstance(item, dict):
            return {"synced": False, "lyrics": "", "provider": ""}
        synced_text = str(item.get("syncedLyrics") or "")
        plain_text = str(item.get("plainLyrics") or "")
        if plain_only:
            plain_output = plain_text or self._strip_lrc_timestamps(synced_text)
            return {
                "synced": False,
                "lyrics": plain_output or "",
                "provider": f"{provider}:plain",
            }
        return {
            "synced": bool(synced_text),
            "lyrics": synced_text or plain_text or "",
            "provider": str(provider or ""),
        }

    # Convert LRC-formatted synced lyrics to plain text for fast/non-timestamped UI mode.
    @staticmethod
    def _strip_lrc_timestamps(text):
        """
        Execute strip lrc timestamps.
        
        This method implements the strip lrc timestamps step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        raw = str(text or "")
        if not raw.strip():
            return ""
        lines = []
        for raw_line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = re.sub(r"\[[0-9]{1,2}:[0-9]{2}(?:[.:][0-9]{1,3})?\]", "", raw_line).strip()
            if not line:
                continue
            lines.append(line)
        deduped = []
        prev = None
        for line in lines:
            if line == prev:
                continue
            deduped.append(line)
            prev = line
        return "\n".join(deduped).strip()

    # Score candidate lyrics matches against requested title and artist.
    def _score_match(self, item, title, artist):
        """
        Score match.
        
        This method implements the score match step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        item_track = self._norm_text(item.get("trackName") or item.get("name") or "")
        item_artist = self._norm_text(item.get("artistName") or "")
        q_track = self._norm_text(title)
        q_artist = self._norm_text(artist)
        score = 0
        if q_track and item_track == q_track:
            score += 6
        elif q_track and q_track in item_track:
            score += 4
        elif q_track and item_track in q_track:
            score += 2
        if q_artist and item_artist == q_artist:
            score += 5
        elif q_artist and q_artist in item_artist:
            score += 3
        elif q_artist and item_artist in q_artist:
            score += 1
        if item.get("syncedLyrics"):
            score += 2
        elif item.get("plainLyrics"):
            score += 1
        return score