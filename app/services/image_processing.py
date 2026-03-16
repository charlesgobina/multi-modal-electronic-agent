import base64
from io import BytesIO

from app.core.config import get_settings


def validate_and_prepare_image(raw: bytes) -> tuple[str, str]:
    settings = get_settings()
    if len(raw) > settings.max_image_bytes:
        raise ValueError(f"Image exceeds {settings.max_image_bytes // (1024 * 1024)} MB limit")

    try:
        from PIL import Image

        image = Image.open(BytesIO(raw))
        image.verify()
        image = Image.open(BytesIO(raw))
    except ImportError as exc:
        raise ValueError("Pillow is required for image preprocessing") from exc
    except Exception as exc:
        raise ValueError(f"Invalid image data: {exc}") from exc

    if max(image.size) > settings.max_image_dim:
        image.thumbnail((settings.max_image_dim, settings.max_image_dim), Image.LANCZOS)

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode(), "image/jpeg"
