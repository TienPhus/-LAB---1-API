import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import base64

MODEL_PATH = "zai-org/GLM-OCR"

# Global model variables - load once at startup
processor = None
model = None


def load_model():
    """Load GLM-OCR model and processor into memory."""
    global processor, model

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.float16,   # float16 for 6-8GB VRAM
        device_map="auto",
    )
    model.eval()
    print("[INFO] Model loaded successfully!")


def is_model_loaded() -> bool:
    return model is not None and processor is not None


def load_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return PIL Image."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def run_ocr(image: Image.Image, mode: str = "text") -> str:
    """
    Run OCR on a PIL Image.
    mode: "text" for simple text recognition, "document" for full document parsing
    """
    if mode == "document":
        prompt_text = "Document Parsing:"
    else:
        prompt_text = "Text Recognition:"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    return output_text.strip()