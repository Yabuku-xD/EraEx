from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.hybrid import searcher
from linkcheck.checker import batch_check_links

BASE_DIR = Path(__file__).parent.parent

app = FastAPI(
    title="ERAEX",
    description="Golden Era(2013-2018)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class SearchRequest(BaseModel):
    query: str
    year_start: Optional[int] = 2012
    year_end: Optional[int] = 2018
    top_k: int = 50


class LuckyRequest(BaseModel):
    history: list = []


class CheckLinksRequest(BaseModel):
    urls: list


@app.on_event("startup")
async def startup():
    print("Loading search indexes...")
    searcher.load()
    print("Ready!")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "healthy", "indexes_loaded": len(searcher.indexes)}


@app.get("/api/search")
async def search_get(
    q: str = Query(..., description="Search query (mood/vibe text)"),
    years: Optional[str] = Query(None, description="Comma-separated years"),
    limit: int = Query(50, ge=1, le=200)
):
    year_list = None
    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    
    results = searcher.search(q, years=year_list, k=limit)
    
    return {"query": q, "results": results, "count": len(results)}


@app.post("/search")
async def search_post(request: SearchRequest):
    if not searcher._loaded:
        return {"error": "System still loading. Please wait.", "retry": True}
    
    years = list(range(request.year_start, request.year_end + 1))
    
    results = searcher.search(request.query, years=years, k=request.top_k)
    
    return {"query": request.query, "results": results, "count": len(results)}


@app.post("/check_links")
async def check_links(request: CheckLinksRequest):
    tracks = []
    for item in request.urls:
        if isinstance(item, dict):
            tracks.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "artist": item.get("artist", "")
            })
        else:
            tracks.append({"url": item, "title": "", "artist": ""})
    
    result_map = await batch_check_links(tracks, max_concurrent=5, searcher=searcher)
    
    return {"results": result_map}


@app.post("/lucky")
async def lucky(request: LuckyRequest):
    if not searcher._loaded:
        return {"error": "System still loading. Please wait.", "retry": True}
    
    history = request.history or []
    
    if not history:
        import random
        vibes = ["chill", "late night", "nostalgic", "summer", "melancholic", "upbeat", "dreamy"]
        combined_query = f"{random.choice(vibes)} vibes"
    else:
        recent = history[:5]
        combined_query = " ".join(recent)
    
    years = list(range(2013, 2019))
    
    results = searcher.search(combined_query, years=years, k=15)
    
    import random
    if len(results) > 10:
        random.shuffle(results)
        results = results[:10]
    
    return {"query": combined_query, "results": results, "count": len(results)}