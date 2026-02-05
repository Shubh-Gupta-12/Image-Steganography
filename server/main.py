from __future__ import annotations

import io
import os
from typing import Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter, ImageEnhance

app = FastAPI(title="ImageStego API - Lightweight")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 ImageStego API starting (lightweight mode)")


def _load_rgb(upload: UploadFile) -> Image.Image:
    """Load image from upload"""
    image_bytes = upload.file.read()
    upload.file.seek(0)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _to_float_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to float array in [0, 1]"""
    return np.asarray(image).astype(np.float32) / 255.0


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert float array to PNG bytes"""
    arr_clipped = np.clip(arr, 0, 1)
    arr_uint8 = (arr_clipped * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resize_match(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Resize b to match a's size"""
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    return a, b


def encode_stegano(cover: np.ndarray, secret: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Blend secret into cover image using alpha blending"""
    return (1 - alpha) * cover + alpha * secret


def decode_stegano(stego: np.ndarray, cover: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Extract secret from stego image"""
    secret = (stego - (1 - alpha) * cover) / alpha
    return np.clip(secret, 0, 1)


def enhance_decoded(arr: np.ndarray) -> np.ndarray:
    """Enhance decoded image for better visibility"""
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    
    # Light blur to reduce noise
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Boost contrast and sharpness
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    
    return np.asarray(img).astype(np.float32) / 255.0


@app.get("/api/health")
def health():
    """Health check"""
    return {"status": "ok", "mode": "lightweight"}


@app.get("/api/debug")
def debug():
    """Debug info for troubleshooting"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    alt_paths = ["/app/server/static", "/app/static", "./static"]
    found_paths = {p: os.path.isdir(p) for p in [static_dir] + alt_paths}
    return {
        "cwd": os.getcwd(),
        "file_location": __file__,
        "static_dir_computed": static_dir,
        "paths_checked": found_paths
    }


@app.post("/api/encode")
def encode(cover: UploadFile = File(...), secret: UploadFile = File(...), alpha: float = Form(0.1)):
    """Encode secret into cover"""
    try:
        if not (0 < alpha <= 1):
            return Response(content=b"Invalid alpha", status_code=400)
        
        cover_img = _load_rgb(cover)
        secret_img = _load_rgb(secret)
        cover_img, secret_img = _resize_match(cover_img, secret_img)
        
        cover_arr = _to_float_array(cover_img)
        secret_arr = _to_float_array(secret_img)
        
        stego_arr = encode_stegano(cover_arr, secret_arr, alpha)
        
        return Response(content=_array_to_png_bytes(stego_arr), media_type="image/png")
    except Exception as e:
        print(f"Encode error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)


@app.post("/api/decode")
def decode(cover: UploadFile = File(...), stego: UploadFile = File(...), alpha: float = Form(0.1)):
    """Decode secret from stego"""
    try:
        if not (0 < alpha <= 1):
            return Response(content=b"Invalid alpha", status_code=400)
        
        cover_img = _load_rgb(cover)
        stego_img = _load_rgb(stego)
        cover_img, stego_img = _resize_match(cover_img, stego_img)
        
        cover_arr = _to_float_array(cover_img)
        stego_arr = _to_float_array(stego_img)
        
        secret_arr = decode_stegano(stego_arr, cover_arr, alpha)
        secret_enhanced = enhance_decoded(secret_arr)
        
        return Response(content=_array_to_png_bytes(secret_enhanced), media_type="image/png")
    except Exception as e:
        print(f"Decode error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)


# Serve frontend static files if available
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
print(f"Looking for static files in: {static_dir}")
print(f"Static dir exists: {os.path.isdir(static_dir)}")

if os.path.isdir(static_dir):
    # List files for debugging
    try:
        files = os.listdir(static_dir)
        print(f"Static files found: {files[:10]}")
    except Exception as e:
        print(f"Error listing static dir: {e}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print("WARNING: Static directory not found, frontend will not be served")
    # Try alternative paths
    alt_paths = ["/app/server/static", "/app/static", "./static"]
    for alt in alt_paths:
        if os.path.isdir(alt):
            print(f"Found static at alternative path: {alt}")
            app.mount("/", StaticFiles(directory=alt, html=True), name="static")
            break

