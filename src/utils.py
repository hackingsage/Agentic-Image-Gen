import io
import zipfile
from PIL import Image

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO() #Create an in-memory byte buffer
    img.save(buf, format=fmt)
    return buf.getvalue()

def zip_images(images):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images, 1):
            zf.writestr(f"image_{i}.png", pil_to_bytes(img))
    buf.seek(0) #Reset buffer position to beginning
    return buf.getvalue()