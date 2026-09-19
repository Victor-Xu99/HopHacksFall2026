import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.hospital_adapt import write_bundle
from src.data.hospital_extract import ingest_and_rank, ingest_extract
from src.data.sql_store import DECISIONS
from src.data.sql_store import available as sql_available
from src.data.sql_store import stay_census
from src.service import (
    SOURCE_HOSPITAL,
    file_review,
    list_sources,
    load_bundle,
    review_payload,
    trained_model,
)

app = FastAPI(title="SafetyNet API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/sources")
def sources():
    return {"sources": list_sources()}


@app.get("/api/review")
def review(
    source: str = Query("safetyhops"),
    model_type: str = Query("logistic"),
    limit: int = Query(20, ge=5, le=100),
):
    available = {item["id"]: item["available"] for item in list_sources()}
    if source not in available:
        raise HTTPException(status_code=400, detail="Unknown data source.")
    if not available[source]:
        raise HTTPException(status_code=400, detail="That data source is not available on this machine.")
    if model_type not in ("logistic", "tree"):
        raise HTTPException(status_code=400, detail="model_type must be logistic or tree.")
    return review_payload(source, model_type, limit)


class ReviewIn(BaseModel):
    source: str
    case_id: str
    decision: str = Field(..., description="harm, no_harm, or unclear")
    reviewer: str = "queue"
    notes: Optional[str] = None


@app.post("/api/reviews")
def save_review(body: ReviewIn):
    if body.decision not in DECISIONS:
        raise HTTPException(status_code=400, detail=f"decision must be one of {DECISIONS}.")
    try:
        review_key = file_review(
            body.source, body.case_id, body.decision, body.reviewer, body.notes
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "review_key": review_key, "case_id": body.case_id, "decision": body.decision}


@app.post("/api/warmup")
def warmup(source: str = Query("safetyhops"), model_type: str = Query("logistic")):
    load_bundle(source)
    trained_model(source, model_type)
    return {"ready": True}


@app.get("/api/hospital/status")
def hospital_status():
    if not sql_available():
        raise HTTPException(status_code=400, detail="SafetyNet SQL is not reachable.")
    return stay_census(SOURCE_HOSPITAL)


@app.post("/api/hospital/ingest")
def hospital_ingest(
    directory: str = Query(..., description="Folder with stays.csv, labs.csv, meds.csv, transfers.csv"),
    rank: bool = Query(True),
    model_type: str = Query("logistic"),
):
    if not sql_available():
        raise HTTPException(status_code=400, detail="SafetyNet SQL is not reachable.")
    path = Path(directory)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {directory}")
    load_bundle.cache_clear()
    trained_model.cache_clear()
    try:
        if rank:
            return ingest_and_rank(path, model_type=model_type)
        return ingest_extract(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/hospital/upload")
async def hospital_upload(
    files: List[UploadFile] = File(..., description="A zip or the CSVs from one folder"),
    rank: bool = Query(True),
    model_type: str = Query("logistic"),
):
    """Import one folder (as many CSVs) or one zip. Headers do not have to match ours."""
    if not sql_available():
        raise HTTPException(status_code=400, detail="SafetyNet SQL is not reachable.")
    if not files:
        raise HTTPException(status_code=400, detail="Upload a folder of CSVs or one zip.")
    folder = Path(tempfile.mkdtemp(prefix="safetynet_extract_"))
    try:
        payload = [(item.filename or "upload.csv", await item.read()) for item in files]
        unpacked = write_bundle(folder, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    load_bundle.cache_clear()
    trained_model.cache_clear()
    try:
        if rank:
            return ingest_and_rank(unpacked, model_type=model_type)
        return ingest_extract(unpacked)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
