import asyncio
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from src.config.settings import BASE_DIR, YEARS
from src.utils.data_loader import data_loader
from src.utils.link_checker import link_checker
from src.search.retrieval import search_years, merge_candidates
from src.search.scorer import score_candidates
from src.search.reranker import rerank


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting data loading in background...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, data_loader.load_all_years)
    yield
    print("Shutting down...")


app = FastAPI(
    title="EraEx Music Recommendation API",
    description="The Golden Era (2012-2018) Music Discovery",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class SearchRequest(BaseModel):
    query: str
    year_start: int = 2012
    year_end: int = 2018
    top_k: int = 10


class LinkCheckRequest(BaseModel):
    urls: List[dict]


class LuckyRequest(BaseModel):
    history: List[str] = []


def format_track(track: dict) -> dict:
    return {
        "id": track.get("id") or track.get("track_id"),
        "title": track.get("title", "Unknown"),
        "artist": track.get("artist", "Unknown"),
        "year": track.get("_year") or track.get("year"),
        "genre": track.get("genre", ""),
        "tags": track.get("tags", ""),
        "permalink_url": track.get("permalink_url", ""),
        "playback_count": track.get("playback_count", 0),
        "score": round(track.get("_final_score", 0), 4),
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search(request: SearchRequest):
    if not data_loader.is_ready():
        return JSONResponse(
            content={"error": "System is still loading data. Please wait a moment.", "retry": True},
            status_code=503
        )
    year_start = max(min(request.year_start, request.year_end), 2012)
    year_end = min(max(request.year_start, request.year_end), 2018)
    if not request.query.strip():
        return JSONResponse(
            content={"error": "Query cannot be empty"},
            status_code=400
        )
    try:
        candidates = search_years(
            query=request.query,
            year_start=year_start,
            year_end=year_end,
            top_k=request.top_k * 5
        )
        unique_candidates = merge_candidates(candidates, top_k=request.top_k * 3)
        scored_candidates = score_candidates(unique_candidates, request.query)
        final_results = rerank(scored_candidates, top_k=request.top_k)
        formatted = [format_track(t) for t in final_results]
        return {
            "query": request.query,
            "year_range": [year_start, year_end],
            "results": formatted,
            "total": len(formatted)
        }
    except Exception as e:
        print(f"Search error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/check_links")
async def check_links(request: LinkCheckRequest):
    try:
        results = await link_checker.check_links(request.urls)
        return {"results": results}
    except Exception as e:
        print(f"Link check error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/lucky")
async def lucky(request: LuckyRequest):
    if not data_loader.is_ready():
        return JSONResponse(
            content={"error": "System is still loading data. Please wait a moment.", "retry": True},
            status_code=503
        )
    if not request.history:
        return JSONResponse(
            content={"error": "No search history found. Try searching first!"},
            status_code=400
        )
    try:
        combined_query = " ".join(request.history[:5])
        selected_years = random.sample(data_loader.get_loaded_years(), min(3, len(data_loader.get_loaded_years())))
        if not selected_years:
            selected_years = [2014, 2015, 2016]
        year_start = min(selected_years)
        year_end = max(selected_years)
        candidates = search_years(
            query=combined_query,
            year_start=year_start,
            year_end=year_end,
            top_k=100
        )
        unique_candidates = merge_candidates(candidates, top_k=50)
        scored = score_candidates(unique_candidates, combined_query)
        if len(scored) > 10:
            selected = random.sample(scored[:30], min(10, len(scored)))
        else:
            selected = scored[:10]
        formatted = [format_track(t) for t in selected]
        return {
            "query": combined_query,
            "results": formatted,
            "total": len(formatted)
        }
    except Exception as e:
        print(f"Lucky error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_years": data_loader.get_loaded_years(),
        "ready": data_loader.is_ready()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)