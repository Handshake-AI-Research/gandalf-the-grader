"""Prepare inline images for OpenAI without modifying workspace evidence.

Resize only enough to meet the rejection limit, preserving detail for judges
that use original resolution. Reject animations rather than discard frames.
"""

import base64
import logging
from io import BytesIO

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
PATCH_EDGE = 32
MAX_PATCHES = 30_000


def _patch_count(width: int, height: int) -> int:
    return ((width + PATCH_EDGE - 1) // PATCH_EDGE) * ((height + PATCH_EDGE - 1) // PATCH_EDGE)


def _fit_size(width: int, height: int) -> tuple[int, int]:
    """Find the largest fitting size for an image already known to exceed the limit."""
    longest = max(width, height)
    low, high = 1, longest - 1
    while low < high:
        candidate = (low + high + 1) // 2
        if _patch_count(max(1, width * candidate // longest), max(1, height * candidate // longest)) <= MAX_PATCHES:
            low = candidate
        else:
            high = candidate - 1
    return max(1, width * low // longest), max(1, height * low // longest)


def prepare_openai_image_url(url: str) -> str:
    """Reject animations and resize oversized images; preserve compliant static bytes."""
    header, separator, encoded = url.partition(",")
    if not separator or not header.lower().startswith("data:") or not header.lower().endswith(";base64"):
        return url

    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(image_bytes)) as source:
        if getattr(source, "is_animated", False):
            msg = "Cannot prepare an animated image for the OpenAI judge without discarding frames."
            raise ValueError(msg)
        width, height = source.size
        if _patch_count(width, height) <= MAX_PATCHES:
            return url

        with ImageOps.exif_transpose(source) as oriented:
            target = _fit_size(*oriented.size)
            has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
            image_format = source.format if source.format in ("JPEG", "WEBP") else "PNG"
            mode = "RGBA" if has_alpha else "RGB"
            save_options = {"quality": 95} if image_format in ("JPEG", "WEBP") else {}
            with oriented.convert(mode) as converted, converted.resize(target, Image.Resampling.LANCZOS) as resized:
                output = BytesIO()
                resized.save(output, format=image_format, **save_options)

    prepared = output.getvalue()
    with Image.open(BytesIO(prepared)) as result:
        if _patch_count(*result.size) > MAX_PATCHES:
            msg = "Prepared image still exceeds the OpenAI patch limit"
            raise ValueError(msg)
    logger.info("Resized judge image from %sx%s to %sx%s for the OpenAI patch limit", width, height, *target)
    mime_type = Image.MIME[image_format]
    return f"data:{mime_type};base64,{base64.b64encode(prepared).decode('ascii')}"
