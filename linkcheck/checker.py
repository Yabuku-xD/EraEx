import httpx
import asyncio
from typing import Optional
import urllib.parse
import re

async def get_client_id(timeout: float = 10.0) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            response = await client.get("https://soundcloud.com/", headers=headers)
            
            if response.status_code == 200:
                scripts = re.findall(r'src="(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"', response.text)
                
                for script_url in scripts[-5:]:
                    try:
                        script_resp = await client.get(script_url, headers=headers)
                        if script_resp.status_code == 200:
                            client_ids = re.findall(r'client_id[:=]["\'"]?([a-zA-Z0-9]{32})["\']?', script_resp.text)
                            if client_ids:
                                return client_ids[0]
                    except:
                        continue
    except Exception as e:
        print(f"Failed to get client_id: {e}")
    
    return None


async def check_soundcloud_oembed(url: str, timeout: float = 5.0) -> dict:
    oembed_url = f"https://soundcloud.com/oembed?url={urllib.parse.quote(url, safe='')}&format=json"
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(oembed_url)
            
            if response.status_code == 200:
                return {"status": "alive", "url": url}
            else:
                return {"status": "dead", "url": url, "reason": "unavailable"}
    except httpx.TimeoutException:
        return {"status": "unknown", "url": url, "reason": "timeout"}
    except Exception as e:
        return {"status": "dead", "url": url, "reason": str(e)}


async def search_soundcloud(query: str, client_id: str, timeout: float = 10.0) -> Optional[str]:
    search_query = " ".join(query.split()[:8])
    search_url = f"https://api-v2.soundcloud.com/search/tracks?q={urllib.parse.quote(search_query)}&client_id={client_id}&limit=10"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(search_url)
            
            if response.status_code == 200:
                data = response.json()
                collection = data.get("collection", [])
                
                for track in collection:
                    permalink = track.get("permalink_url")
                    if permalink:
                        return permalink
    except Exception as e:
        print(f"Search failed: {e}")
    
    return None


_cached_client_id = None

async def get_cached_client_id() -> Optional[str]:
    global _cached_client_id
    if _cached_client_id is None:
        _cached_client_id = await get_client_id()
    return _cached_client_id


async def check_and_heal_link(url: str, title: str, artist: str) -> dict:
    check_result = await check_soundcloud_oembed(url)
    
    if check_result["status"] == "alive":
        return check_result
    
    client_id = await get_cached_client_id()
    
    if client_id:
        search_queries = [
            f"{artist} {title}",
            title,
        ]
        
        for query in search_queries:
            if not query or len(query.strip()) < 3:
                continue
            
            alternative = await search_soundcloud(query, client_id)
            
            if alternative and alternative != url:
                verify = await check_soundcloud_oembed(alternative)
                if verify["status"] == "alive":
                    return {
                        "status": "healed",
                        "original_url": url,
                        "new_url": alternative,
                        "reason": "found_alternative"
                    }
    
    return {
        "status": "dead",
        "url": url,
        "reason": check_result.get("reason", "unknown")
    }


async def batch_check_links(tracks: list, max_concurrent: int = 5, searcher=None) -> dict:
    print("Getting SoundCloud client_id...")
    client_id = await get_cached_client_id()
    print(f"Client ID: {client_id}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_with_semaphore(track):
        async with semaphore:
            url = track.get("url", "")
            title = track.get("title", "")
            artist = track.get("artist", "")
            result = await check_and_heal_link(url, title, artist)
            return url, result
    
    tasks = [check_with_semaphore(t) for t in tracks]
    results = await asyncio.gather(*tasks)
    
    return {url: result for url, result in results}