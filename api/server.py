"""
api/server.py  —  CricketLytics FastAPI backend

Run from the repo root:
    uvicorn api.server:app --reload --port 8000

The dashboard static files are served at / from the dashboard/ directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="CricketLytics API",
    description="AI-powered cricket batting analysis backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_HANDS = {"left", "right"}
VALID_VIEWS = {"front", "side"}
VALID_SHOTS = {"auto", "drive", "defensive", "pull", "sweep", "cut", "back-punch"}


# ── Analysis endpoint ──────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(..., description="Cricket video file (MP4 / AVI)"),
    hand: str        = Form(..., description="'left' or 'right'"),
    view: str        = Form(..., description="'front' or 'side'"),
    intended_shot: str = Form(default="auto", description="Shot type or 'auto'"),
) -> JSONResponse:
    """
    Accept an uploaded video and analysis parameters, run the headless
    batting analyser, and return aggregated statistics as JSON.
    Annotated preview frames (base64 JPEG) are embedded in the response
    under the key 'preview_frames'.
    """

    # ── Validate parameters ───────────────────────────────────────────
    if hand.lower() not in VALID_HANDS:
        raise HTTPException(400, f"hand must be one of {VALID_HANDS}")
    if view.lower() not in VALID_VIEWS:
        raise HTTPException(400, f"view must be one of {VALID_VIEWS}")
    if intended_shot.lower() not in VALID_SHOTS:
        raise HTTPException(400, f"intended_shot must be one of {VALID_SHOTS}")

    # ── Save uploaded video to a temp file ────────────────────────────
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            shutil.copyfileobj(video.file, fh)

        # ── Run headless analysis ──────────────────────────────────────
        from src.Analysis.headless_analyzer import HeadlessAnalyzer

        analyzer = HeadlessAnalyzer()
        result   = analyzer.analyze(tmp_path, hand, view, intended_shot)

    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return JSONResponse(content=result)


# ── Health check ───────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "CricketLytics API"}


# ── Serve dashboard static files ───────────────────────────────────────
_dashboard_dir = Path(__file__).parent.parent / "dashboard"
if _dashboard_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_dashboard_dir), html=True),
        name="dashboard",
    )
