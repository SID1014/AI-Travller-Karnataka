"""
main.py - FastAPI app. One process serves both the API (/api/...) and the web page (/), so there is
a single URL to deploy and no CORS to fight.

Run locally (from the backend folder):
    uvicorn main:app --reload --port 8000
then open http://localhost:8000
"""

import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")            # project-root .env
load_dotenv(Path(__file__).parent / ".env")   # or backend/.env

from service import PlannerService  # noqa: E402  (after env is loaded so LLM settings are visible)

DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "data" / "places_v1.csv"))
STAYS_PATH = os.getenv("STAYS_PATH", str(ROOT / "data" / "city_stay_costs.csv"))
FRONTEND_DIR = ROOT / "frontend"

state = {}


@asynccontextmanager
async def lifespan(app):
    state["svc"] = PlannerService(DATA_PATH, STAYS_PATH)
    yield


app = FastAPI(title="Karnataka Travel Planner", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class PlanRequest(BaseModel):
    days: int | None = Field(default=None, ge=1, le=7)
    interests: list[str] = []
    month: int | None = Field(default=None, ge=1, le=12)
    pace: str | None = None
    start_cluster: str | None = None
    travelers: int | None = Field(default=None, ge=1, le=12)
    budget_inr: int | None = Field(default=None, ge=0)
    stay_type: str | None = None
    free_text: str = Field(default="", max_length=500)
    use_llm: bool = True


@app.get("/api/health")
def health():
    svc = state.get("svc")
    return {"ok": svc is not None, "places": len(svc.places) if svc else 0,
            "llm": svc.llm.label if svc else "off"}


@app.get("/api/meta")
def meta():
    return state["svc"].meta()


@app.post("/api/plan")
def plan(req: PlanRequest):
    # plain `def` so FastAPI runs it in a worker thread (the LLM calls block)
    try:
        return state["svc"].plan(req.model_dump())
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Could not build a plan. Try fewer days or different interests.")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")