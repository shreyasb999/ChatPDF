# processors/image_utils.py

from typing import List
import base64
from PIL import Image
from io import BytesIO

def extract_base64_images(image_chunks) -> List[str]:
    """
    From image chunks, collect their base64 strings.
    """
    images = []
    for chunk in image_chunks:
        # each chunk may carry a payload of an ImageElement
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "image_base64"):
            images.append(chunk.metadata.image_base64)
    return images

def b64_to_pil(image_b64: str) -> Image.Image:
    """
    Convert a base64 string into a PIL Image.
    """
    data = base64.b64decode(image_b64)
    return Image.open(BytesIO(data))
