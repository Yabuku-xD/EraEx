import asyncio
import aiohttp
from typing import Dict, List, Optional
from urllib.parse import quote_plus


class LinkChecker:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._timeout = aiohttp.ClientTimeout(total=5)

    async def check_single_link(
        self,
        session: aiohttp.ClientSession,
        url: str,
        artist: str = "",
        title: str = ""
    ) -> dict:
        if url in self._cache:
            return self._cache[url]
        try:
            async with session.head(url, allow_redirects=True) as response:
                if response.status == 200:
                    result = {"status": "alive", "url": url}
                elif response.status in [301, 302, 303, 307, 308]:
                    new_url = str(response.url)
                    result = {"status": "healed", "url": url, "new_url": new_url}
                else:
                    result = await self._try_youtube_fallback(artist, title, url)
        except asyncio.TimeoutError:
            result = await self._try_youtube_fallback(artist, title, url)
        except Exception:
            result = await self._try_youtube_fallback(artist, title, url)
        self._cache[url] = result
        return result

    async def _try_youtube_fallback(
        self,
        artist: str,
        title: str,
        original_url: str
    ) -> dict:
        if artist or title:
            search_query = f"{title} {artist}".strip()
            yt_search_url = f"https://www.youtube.com/results?search_query={quote_plus(search_query)}"
            return {
                "status": "replaced",
                "url": original_url,
                "new_url": yt_search_url,
                "new_track": {"title": title, "artist": artist}
            }
        return {"status": "dead", "url": original_url}

    async def check_links(
        self,
        urls: List[dict]
    ) -> Dict[str, dict]:
        results = {}
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            tasks = []
            for item in urls:
                url = item.get("url", "")
                artist = item.get("artist", "")
                title = item.get("title", "")
                tasks.append(self.check_single_link(session, url, artist, title))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for item, response in zip(urls, responses):
                url = item.get("url", "")
                if isinstance(response, Exception):
                    results[url] = {"status": "dead", "url": url}
                else:
                    results[url] = response
        return results

    def clear_cache(self):
        self._cache.clear()


link_checker = LinkChecker()
