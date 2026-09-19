from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.service import list_sources, review_payload, trained_model, load_bundle

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


@app.post("/api/warmup")
def warmup(source: str = Query("safetyhops"), model_type: str = Query("logistic")):
    load_bundle(source)
    trained_model(source, model_type)
    return {"ready": True}
