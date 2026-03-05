import os
import shutil
import uuid
import logging
# FOR NORMAL STABLE USAGE, RUN: .\start_app.ps1
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
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
except ImportError as e:
    logger.error(f"Could not import backend modules: {e}")
    raise RuntimeError(f"Could not import backend modules: {e}")

# Global state for ML model
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML models and resources on startup.
    Clean them up on shutdown.
    """
    logger.info("Starting up: Loading ViT model and index...")
    try:
        processor, model = load_vit_model()
        # Ensure embedding file exists or warn
        if not os.path.exists(EMBEDDING_FILE):
             logger.warning(f"Embedding file '{EMBEDDING_FILE}' not found. Predictions might fail.")
             index, embeddings, meta = None, None, None
        else:
             index, embeddings, meta = load_index_and_meta(EMBEDDING_FILE)
        
        ml_resources["processor"] = processor
        ml_resources["model"] = model
        ml_resources["index"] = index
        ml_resources["meta"] = meta
        logger.info("ViT model and index loaded successfully!")
    except Exception as e:
        logger.error(f"Error during startup resource loading: {e}")
        # In a real app, we might want to shut down if the model fails to load
        # but for this project we'll allow startup so /health works.
    
    yield
    
    # Cleanup
    ml_resources.clear()
    logger.info("Shutting down... resources cleared.")

app = FastAPI(
    title="Snake Bite Detection API",
    description="API for identifying snake species from images using ViT.",
    version="1.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount Static Files (Frontend)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Pydantic Models
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

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    """
    Predict snake species from an uploaded image.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Validate content type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG/PNG allowed.")

    # Generate unique filename
    file_ext = image.filename.split(".")[-1]
    filename = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        # Save file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Check if model is loaded
        if not ml_resources.get("model") or not ml_resources.get("index"):
             logger.error("Predict called but model/index not loaded.")
             raise HTTPException(
                 status_code=500, 
                 detail="Model not loaded. Please check server logs."
             )

        # Run prediction
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
            raise HTTPException(status_code=500, detail="No prediction returned from backend.")

        top_match = results[0]

        # Map backend result to response model
        response = PredictionResponse(
            species_name=top_match.get("species_name", "Unknown"),
            venom_type=top_match.get("venom_type", "Unknown"),
            symptoms=top_match.get("symptoms", "Not available"),
            reaction_stage=top_match.get("reaction_stage", "Not available"),
            severity_level=top_match.get("severity_level", "Unknown"),
            first_aid=top_match.get("first_aid", "Not available"),
            hospital_importance=top_match.get("hospital_importance", "Not specified"),
            similarity_score=top_match.get("similarity_score", 0.0)
        )
        
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {file_path}: {e}")

    import uvicorn
    # ── NEW ENDPOINT: /chat ───────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Snakebite emergency chatbot.
    Accepts natural language queries, returns structured medical guidance.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty."
        )
    if len(request.message) > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message too long (max 1000 characters)."
        )
    try:
        response = process_chat(request)
        return response
    except Exception as e:
        logger.exception("Chatbot error")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


# ── NEW ENDPOINT: /nearby-hospitals ───────────────────────────────────────────
@app.post("/nearby-hospitals", response_model=HospitalResponse)
async def nearby_hospitals(request: HospitalRequest):
    """
    Find hospitals near given GPS coordinates.
    Uses OpenStreetMap (free) by default, or Google Places if provider='google'.
    """
    try:
        result = await find_nearby_hospitals(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Hospital finder error")
        raise HTTPException(status_code=500, detail=f"Hospital search failed: {str(e)}")
   

