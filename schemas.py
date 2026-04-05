from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class OCRMode(str, Enum):
    text = "text"           # Simple text recognition
    document = "document"   # Full document parsing (tables, formulas, layout)


class PredictURLRequest(BaseModel):
    url: str = Field(..., description="URL of the image to process")
    mode: OCRMode = Field(
        default=OCRMode.text,
        description="OCR mode: 'text' for simple recognition, 'document' for full parsing"
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class PredictResponse(BaseModel):
    success: bool
    mode: str
    result: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None