import os
import shutil
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.chatbot import ChatRequest, ChatResponse, process_chat
from src.hospital_locator import HospitalRequest, HospitalResponse, find_nearby_hospitals

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SnakeGuard")

# Import existing backend logic
try:
    from vit_snake_detection import (
        load_vit_model,
        load_index_and_meta,
        predict_one_image,
        DEFAULT_CSV_PATH,
        EMBEDDING_FILE
    )
    logger.info("Backend modules imported successfully.")
except ImportError as e:
    logger.error(f"Could not import backend modules: {e}")
    raise RuntimeError(f"Could not import backend modules: {e}")

# Global state for ML model
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Loading ViT model and index...")
    try:
        processor, model = load_vit_model()

        if not os.path.exists(EMBEDDING_FILE):
            logger.warning(f"Embedding file '{EMBEDDING_FILE}' not found. /predict will fail.")
            index, embeddings, meta = None, None, None
        else:
            logger.info(f"Loading embeddings from: {EMBEDDING_FILE}")
            index, embeddings, meta = load_index_and_meta(EMBEDDING_FILE)
            logger.info(f"Index loaded. Meta keys: {list(meta.keys()) if meta else 'None'}")

        ml_resources["processor"] = processor
        ml_resources["model"] = model
        ml_resources["index"] = index
        ml_resources["meta"] = meta
        logger.info("ViT model and index loaded successfully!")

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)

    yield

    ml_resources.clear()
    logger.info("Shutting down... resources cleared.")


app = FastAPI(
    title="Snake Bite Detection API",
    description="API for identifying snake species from images using ViT.",
    version="1.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    species_name: str
    venom_type: str
    symptoms: str
    reaction_stage: str
    severity_level: str
    first_aid: str
    hospital_importance: str
    similarity_score: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    csv_exists: bool
    embedding_file_exists: bool


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model_loaded": ml_resources.get("model") is not None,
        "index_loaded": ml_resources.get("index") is not None,
        "csv_exists": os.path.exists(DEFAULT_CSV_PATH),
        "embedding_file_exists": os.path.exists(EMBEDDING_FILE),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    """
    Predict snake species from an uploaded image.
    Accepts field name 'image' (multipart/form-data).
    """
    logger.info(f"Predict called — filename={image.filename}, content_type={image.content_type}")

    # ── Validate ──
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "application/octet-stream"]
    if image.content_type not in allowed_types:
        logger.warning(f"Rejected content_type: {image.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid image type '{image.content_type}'. Use JPEG or PNG.")

    # ── Check resources ──
    if ml_resources.get("model") is None:
        logger.error("Model not loaded in ml_resources.")
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    if ml_resources.get("index") is None:
        logger.error("Index not loaded — embedding file may be missing.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Embedding index not loaded. "
                f"Run: python vit_snake_detection.py --build-index --csv \"{DEFAULT_CSV_PATH}\" "
                "to generate the index first."
            )
        )

    if not os.path.exists(DEFAULT_CSV_PATH):
        logger.error(f"CSV not found at: {DEFAULT_CSV_PATH}")
        raise HTTPException(
            status_code=500,
            detail=f"Metadata CSV not found at: {DEFAULT_CSV_PATH}"
        )

    # ── Save temp file ──
    file_ext = (image.filename.split(".")[-1] if "." in image.filename else "jpg")
    filename = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(image.file, buf)
        logger.info(f"Image saved: {file_path} ({os.path.getsize(file_path)} bytes)")

        # ── Run prediction ──
        results = predict_one_image(
            image_path=file_path,
            processor=ml_resources["processor"],
            model=ml_resources["model"],
            index=ml_resources["index"],
            meta=ml_resources["meta"],
            k=1,
            csv_path=DEFAULT_CSV_PATH
        )

        if not results:
            raise HTTPException(status_code=500, detail="No prediction returned.")

        top = results[0]
        logger.info(f"Prediction result: {top.get('species_name')} | score={top.get('similarity_score'):.4f}")

        return PredictionResponse(
            species_name=top.get("species_name", "Unknown"),
            venom_type=top.get("venom_type", "Unknown"),
            symptoms=top.get("symptoms", "Not available"),
            reaction_stage=top.get("reaction_stage", "Not available"),
            severity_level=top.get("severity_level", "Unknown"),
            first_aid=top.get("first_aid", "Not available"),
            hospital_importance=top.get("hospital_importance", "Not specified"),
            similarity_score=float(top.get("similarity_score", 0.0))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if len(request.message) > 1000:
        raise HTTPException(status_code=422, detail="Message too long (max 1000 chars).")
    try:
        return process_chat(request)
    except Exception as e:
        logger.exception("Chatbot error")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


@app.post("/nearby-hospitals", response_model=HospitalResponse)
async def nearby_hospitals(request: HospitalRequest):
    try:
        return await find_nearby_hospitals(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Hospital finder error")
        raise HTTPException(status_code=500, detail=f"Hospital search failed: {str(e)}")


# ── Debug endpoint — helps diagnose predict issues ───────────────────────────
@app.get("/debug")
async def debug_info():
    """Check all paths and resource states."""
    return {
        "cwd": os.getcwd(),
        "embedding_file": EMBEDDING_FILE,
        "embedding_exists": os.path.exists(EMBEDDING_FILE),
        "csv_path": DEFAULT_CSV_PATH,
        "csv_exists": os.path.exists(DEFAULT_CSV_PATH),
        "upload_dir": UPLOAD_DIR,
        "upload_dir_exists": os.path.exists(UPLOAD_DIR),
        "model_loaded": ml_resources.get("model") is not None,
        "index_loaded": ml_resources.get("index") is not None,
        "meta_keys": list(ml_resources["meta"].keys()) if ml_resources.get("meta") else None,
        "static_dir_exists": os.path.exists("static"),
        "index_html_exists": os.path.exists("static/index.html"),
    }
