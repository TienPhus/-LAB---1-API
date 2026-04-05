import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO

from schemas import (
    PredictURLRequest,
    PredictResponse,
    OCRMode,
)
import model as ocr_model


# ── Lifespan: load model once when server starts ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading GLM-OCR model...")
    ocr_model.load_model()
    yield
    print("[SHUTDOWN] Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GLM-OCR API",
    description=(
        "OCR API powered by zai-org/GLM-OCR — a 0.9B multimodal model "
        "for complex document understanding. Supports image upload and URL input."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/", summary="API Introduction")
def root():
    return {
        "name": "GLM-OCR API",
        "version": "1.0.0",
        "model": "zai-org/GLM-OCR",
        "description": "Extract text from images using GLM-OCR",
        "endpoints": {
            "GET  /": "API introduction",
            "GET  /health": "Health check",
            "POST /predict/url": "OCR from image URL",
            "POST /predict/upload": "OCR from uploaded file",
        },
        "docs": "/docs",
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Health Check")
def health():
    loaded = ocr_model.is_model_loaded()
    device = "N/A"
    if loaded:
        try:
            device = str(next(ocr_model.model.parameters()).device)
        except Exception:
            device = "unknown"
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "device": device,
        "message": "Model is ready." if loaded else "Model is not loaded yet.",
    }


# ── POST /predict/upload ──────────────────────────────────────────────────────
@app.post("/predict/upload", response_model=PredictResponse, summary="OCR from uploaded image file")
async def predict_upload(
    file: UploadFile = File(..., description="Image file (jpg, png)"),
    mode: OCRMode = Form(default=OCRMode.text, description="OCR mode: text or document"),
):
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Allowed: jpg, png",
        )

    if not ocr_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = ocr_model.run_ocr(image, mode=mode.value)
        return PredictResponse(
            success=True,
            mode=mode.value,
            result=result,
            message=f"OCR completed on uploaded file: {file.filename}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


# ── POST /predict/url ─────────────────────────────────────────────────────────
@app.post("/predict/url", response_model=PredictResponse, summary="OCR from image URL")
def predict_url(request: PredictURLRequest):
    if not ocr_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        image = ocr_model.load_image_from_url(request.url)
        result = ocr_model.run_ocr(image, mode=request.mode.value)
        return PredictResponse(
            success=True,
            mode=request.mode.value,
            result=result,
            message="OCR completed from URL.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )