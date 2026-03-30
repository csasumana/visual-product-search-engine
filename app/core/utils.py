from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_extension(filename: Optional[str]) -> None:
    if not filename:
        return
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


async def read_upload_as_pil(upload_file: UploadFile) -> Image.Image:
    validate_image_extension(upload_file.filename)

    try:
        contents = await upload_file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded image: {str(e)}")


def load_local_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load local image {image_path}: {str(e)}")


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0