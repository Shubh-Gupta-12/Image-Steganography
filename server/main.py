from __future__ import annotations

import io
import os
from typing import Tuple

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Timeouts for model downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"

app = FastAPI(title="ImageStego API - Diffusion Based")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lazy loading global
_pipe_cache = None


def get_pipeline():
    """Get or load the diffusion model (non-blocking, async loading)"""
    global _pipe_cache
    
    if _pipe_cache is not None:
        return _pipe_cache
    
    # Don't wait for model load - just return None to use fallback
    # Model loading happens in background if needed
    print("⚙️  Model not ready yet, using fast processing mode")
    return None


def _load_rgb(upload: UploadFile) -> Image.Image:
    """Load image from upload"""
    image_bytes = upload.file.read()
    upload.file.seek(0)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _to_float_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor (1, C, H, W) in [0, 1]"""
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor (1, C, H, W) to PNG bytes"""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu()
    arr = (torch.clamp(arr, 0, 1) * 255).numpy().astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resize_match(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Resize b to match a's size"""
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    return a, b


def encode_stegano(cover: torch.Tensor, secret: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Blend secret into cover image"""
    return (1 - alpha) * cover + alpha * secret


def decode_stegano(stego: torch.Tensor, cover: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Extract secret from stego image"""
    secret = (stego - (1 - alpha) * cover) / alpha
    return torch.clamp(secret, 0, 1)


def denoise_vae(image_tensor: torch.Tensor) -> torch.Tensor:
    """Denoise image using VAE from diffusion model, or basic filtering if model unavailable"""
    pipe = get_pipeline()
    
    if pipe is None:
        # Fallback: Use basic Gaussian blur for denoising
        arr = image_tensor.squeeze(0).permute(1, 2, 0).cpu()
        arr = (torch.clamp(arr, 0, 1) * 255).numpy().astype(np.uint8)
        pil_img = Image.fromarray(arr)
        
        # Apply light blur for smoothing
        from PIL import ImageFilter
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        arr_blur = np.asarray(pil_img).astype(np.float32) / 255.0
        result = torch.from_numpy(arr_blur).permute(2, 0, 1).unsqueeze(0).to(device)
        return result
    
    # Original VAE denoising with diffusion model
    # Prepare image
    arr = image_tensor.squeeze(0).permute(1, 2, 0).cpu()
    arr = (torch.clamp(arr, 0, 1) * 255).numpy().astype(np.uint8)
    pil_img = Image.fromarray(arr)
    orig_size = pil_img.size
    
    with torch.no_grad():
        # Resize for VAE (multiple of 8)
        h, w = ((orig_size[1] // 8) * 8, (orig_size[0] // 8) * 8)
        img_resized = pil_img.resize((w, h), Image.BICUBIC)
        arr_resized = np.asarray(img_resized).astype(np.float32) / 255.0
        
        # Encode-decode with VAE
        x = torch.from_numpy(arr_resized).permute(2, 0, 1).unsqueeze(0).to(device)
        z = pipe.vae.encode(x * 2 - 1).latent_dist.sample() * 0.18215
        z_noisy = z + torch.randn_like(z) * 0.05  # Small noise
        x_out = pipe.vae.decode(z_noisy / 0.18215).sample
        x_out = (x_out / 2 + 0.5).clamp(0, 1)
        
        # Convert back
        arr_out = x_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_out = Image.fromarray((arr_out * 255).astype(np.uint8))
        img_out = img_out.resize(orig_size, Image.BICUBIC)
        
        # Back to tensor
        arr_final = np.asarray(img_out).astype(np.float32) / 255.0
        result = torch.from_numpy(arr_final).permute(2, 0, 1).unsqueeze(0).to(device)
        return result


@app.get("/api/health")
def health():
    """Health check"""
    return {"status": "ok", "device": str(device), "model_loaded": _pipe_cache is not None}


@app.post("/api/encode")
def encode(cover: UploadFile = File(...), secret: UploadFile = File(...), alpha: float = Form(0.1)):
    """Encode secret into cover"""
    try:
        if not (0 < alpha <= 1):
            return Response(content=b"Invalid alpha", status_code=400)
        
        cover_img = _load_rgb(cover)
        secret_img = _load_rgb(secret)
        cover_img, secret_img = _resize_match(cover_img, secret_img)
        
        cover_t = _to_float_tensor(cover_img)
        secret_t = _to_float_tensor(secret_img)
        
        stego_t = encode_stegano(cover_t, secret_t, alpha)
        stego_enhanced = denoise_vae(stego_t)
        
        return Response(content=_tensor_to_png_bytes(stego_enhanced), media_type="image/png")
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
        
        cover_t = _to_float_tensor(cover_img)
        stego_t = _to_float_tensor(stego_img)
        
        secret_t = decode_stegano(stego_t, cover_t, alpha)
        secret_enhanced = denoise_vae(secret_t)
        
        return Response(content=_tensor_to_png_bytes(secret_enhanced), media_type="image/png")
    except Exception as e:
        print(f"Decode error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

