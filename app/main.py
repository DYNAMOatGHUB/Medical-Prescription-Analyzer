"""
app/main.py
────────────
FastAPI backend for the Medical Prescription Analyser.

Endpoints
---------
  GET  /                 Health check / welcome
  GET  /models/status    Active VLM configuration
  POST /predict          Upload image → VLM extraction → return structured JSON
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.pipeline.vlm_extractor import extract_prescription

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Prescription Analyser",
    description=(
        "Vision-Language Model pipeline that reads handwritten medical "
        "prescriptions and extracts structured medicine data via Ollama."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "Medical Prescription Analyser",
        "version": "2.0.0",
    }


@app.get("/models/status", tags=["Health"])
async def model_status() -> Dict[str, Any]:
    """Return active model configuration."""
    return {
        "vlm_model": settings.VLM_MODEL,
        "log_level": settings.LOG_LEVEL,
    }


@app.post("/predict", tags=["Pipeline"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a prescription image (JPG / PNG / BMP / TIFF / WebP) and get back
    a structured JSON with extracted medicine names, dosages, and schedules.
    """
    # ── Validate content type ──────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Please upload a JPG, PNG, BMP, or TIFF image.",
        )

    t_start = time.perf_counter()
    raw_bytes = await file.read()

    try:
        logger.info(f"[predict] Routing image '{file.filename}' to VLM Extractor ...")
        prescription_data = extract_prescription(raw_bytes, mime_type=file.content_type)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    elapsed = round(time.perf_counter() - t_start, 3)
    logger.info(f"[predict] Done in {elapsed}s")
    logger.info(f"⚡ [PERFORMANCE] Total API Endpoint Latency: {elapsed:.3f} seconds")

    return JSONResponse({
        "filename": file.filename,
        "processing_time_seconds": elapsed,
        "prescription": prescription_data.model_dump(),
    })
