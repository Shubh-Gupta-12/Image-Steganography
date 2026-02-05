# -*- coding: utf-8 -*-
"""Simple image-in-image steganography demo.

Usage:
  - Set FLICKR8K_IMAGES_PATH env var to a folder of .jpg images, or
  - Place images in data/Flickr8k/Images.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16


device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


class Flickr8kDataset(Dataset):
    def __init__(self, images_path: str, transform=None):
        super().__init__()
        self.images_path = Path(images_path)
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(images_path) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = self.images_path / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


def encode_stegano(cover: torch.Tensor, secret: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Encodes secret into cover image using weighted blend."""
    return (1 - alpha) * cover + alpha * secret


def decode_stegano(encoded_image: torch.Tensor, cover_image: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Extracts the secret image from the encoded image using inverse blending."""
    if encoded_image.shape != cover_image.shape:
        raise ValueError("Encoded and Cover images must have the same shape!")
    secret_recovered = (encoded_image - (1 - alpha) * cover_image) / alpha
    return torch.clamp(secret_recovered, 0, 1)


def show_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


def calculate_metrics(original_image: torch.Tensor, decoded_image: torch.Tensor):
    """Calculates SSIM and MSE between original and decoded images."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse

    original_np = (original_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    decoded_np = (decoded_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    original_gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    decoded_gray = cv2.cvtColor(decoded_np, cv2.COLOR_RGB2GRAY)

    ssim_index = ssim(original_gray, decoded_gray)
    mean_squared_error = mse(original_np, decoded_np)

    return ssim_index, mean_squared_error


def get_vit_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ViT training/inference.")
    return torch.device("cuda")


VIT_IMAGE_SIZE = 224

vit_base_transform = transforms.Compose([
    transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
    transforms.ToTensor(),
])

vit_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class StegoVitDataset(Dataset):
    def __init__(self, images_path: str, alpha: float = 0.1, stego_prob: float = 0.5):
        super().__init__()
        self.images_path = Path(images_path)
        self.alpha = alpha
        self.stego_prob = stego_prob
        self.image_filenames = [f for f in os.listdir(images_path) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def _load_image(self, filename: str) -> torch.Tensor:
        img_path = self.images_path / filename
        image = Image.open(img_path).convert("RGB")
        return vit_base_transform(image)

    def __getitem__(self, idx):
        cover_name = self.image_filenames[idx]
        secret_name = random.choice(self.image_filenames)

        cover = self._load_image(cover_name)
        secret = self._load_image(secret_name)

        if random.random() < self.stego_prob:
            stego = encode_stegano(cover, secret, alpha=self.alpha)
            label = torch.tensor(1.0)
            target = secret
            return vit_normalize(stego), target, label

        label = torch.tensor(0.0)
        target = torch.zeros_like(cover)
        return vit_normalize(cover), target, label


class ViTStegoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()

        self.classifier = nn.Linear(768, 1)
        self.decoder = nn.Sequential(
            nn.Linear(768, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)
        decoded = self.decoder(features)
        return logits, decoded


def train_vit(images_path: str, model_path: str, epochs: int = 5, batch_size: int = 8, alpha: float = 0.1):
    vit_device = get_vit_device()

    dataset = StegoVitDataset(images_path, alpha=alpha)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 images in the dataset folder.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    total_batches = len(loader)

    model = ViTStegoModel().to(vit_device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    recon_weight = float(os.getenv("VIT_RECON_WEIGHT", "5.0"))

    log_every = int(os.getenv("VIT_LOG_EVERY", "50"))
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} - batches: {total_batches}")
        total_loss = 0.0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            inputs, targets, labels = batch
            inputs = inputs.to(vit_device, non_blocking=True)
            targets = targets.to(vit_device, non_blocking=True)
            labels = labels.to(vit_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, decoded = model(inputs)

            cls_loss = bce(logits, labels)

            mask = labels.view(-1, 1, 1, 1)
            recon_diff = torch.abs(decoded - targets)
            denom = mask.sum() * recon_diff[0].numel()
            if denom > 0:
                recon_loss = (recon_diff * mask).sum() / denom
            else:
                recon_loss = torch.zeros((), device=vit_device)

            loss = cls_loss + recon_weight * recon_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - epoch_start
                avg_time = elapsed / max(1, batch_idx)
                remaining = max(0, total_batches - batch_idx)
                eta_sec = remaining * avg_time
                print(
                    f"  Batch {batch_idx}/{total_batches} - loss: {loss.item():.4f} - ETA: {eta_sec:.1f}s"
                )

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved ViT model to {model_path}")


def decode_with_vit(model_path: str, image_path: str, output_path: str):
    vit_device = get_vit_device()
    model = ViTStegoModel().to(vit_device)
    model.load_state_dict(torch.load(model_path, map_location=vit_device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    x = vit_normalize(vit_base_transform(image)).unsqueeze(0).to(vit_device)

    with torch.no_grad():
        logits, decoded = model(x)
        prob = torch.sigmoid(logits).item()

        decoded = F.interpolate(
            decoded,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

    decoded_img = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded_img = np.clip(decoded_img, 0, 1)

    # Post-process to make the output more visible
    if os.getenv("VIT_POSTPROCESS", "1") == "1":
        flat = decoded_img.reshape(-1, 3)
        lo = np.percentile(flat, 0.5, axis=0)
        hi = np.percentile(flat, 99.5, axis=0)
        decoded_img = (decoded_img - lo) / (hi - lo + 1e-6)
        decoded_img = np.clip(decoded_img, 0, 1)
        gamma = float(os.getenv("VIT_GAMMA", "0.6"))
        decoded_img = np.power(decoded_img, gamma)

        if os.getenv("VIT_SHARPEN", "1") == "1":
            img_8 = (decoded_img * 255).astype(np.uint8)
            blur = cv2.GaussianBlur(img_8, (0, 0), 1.0)
            sharpened = cv2.addWeighted(img_8, 1.4, blur, -0.4, 0)
            decoded_img = np.clip(sharpened.astype(np.float32) / 255.0, 0, 1)

    decoded_img = np.clip(decoded_img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(decoded_img).save(output_path)

    print(f"Stego probability: {prob:.4f}")
    print(f"Decoded output saved to {output_path}")


def evaluate_vit(images_path: str, model_path: str, samples: int = 512, batch_size: int = 8, alpha: float = 0.1):
    vit_device = get_vit_device()
    model = ViTStegoModel().to(vit_device)
    model.load_state_dict(torch.load(model_path, map_location=vit_device))
    model.eval()

    dataset = StegoVitDataset(images_path, alpha=alpha)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 images in the dataset folder.")

    if samples > len(dataset):
        samples = len(dataset)

    subset_indices = random.sample(range(len(dataset)), samples)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    bce = nn.BCEWithLogitsLoss(reduction="sum")
    recon_weight = float(os.getenv("VIT_RECON_WEIGHT", "5.0"))

    total_loss = 0.0
    total_cls_correct = 0
    total_cls = 0

    with torch.no_grad():
        for inputs, targets, labels in loader:
            inputs = inputs.to(vit_device, non_blocking=True)
            targets = targets.to(vit_device, non_blocking=True)
            labels = labels.to(vit_device, non_blocking=True)

            logits, decoded = model(inputs)
            cls_loss = bce(logits, labels)

            mask = labels.view(-1, 1, 1, 1)
            recon_diff = torch.abs(decoded - targets)
            denom = mask.sum() * recon_diff[0].numel()
            if denom > 0:
                recon_loss = (recon_diff * mask).sum() / denom
            else:
                recon_loss = torch.zeros((), device=vit_device)

            batch_loss = cls_loss + recon_weight * recon_loss * labels.size(0)
            total_loss += batch_loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_cls_correct += (preds == labels).sum().item()
            total_cls += labels.numel()

    avg_loss = total_loss / max(1, total_cls)
    accuracy = total_cls_correct / max(1, total_cls)
    print(f"Eval samples: {samples}")
    print(f"Detection accuracy: {accuracy:.4f}")
    print(f"Avg loss per sample: {avg_loss:.4f}")


if __name__ == "__main__":
    if os.getenv("VIT_TRAIN", "0") == "1":
        images_path = os.getenv("FLICKR8K_IMAGES_PATH", "data/Flickr8k/Images")
        if not os.path.isdir(images_path):
            raise FileNotFoundError(
                "Images folder not found. Set FLICKR8K_IMAGES_PATH env var or place images in data/Flickr8k/Images."
            )
        model_path = os.getenv("VIT_MODEL_PATH", "vit_stego.pth")
        epochs = int(os.getenv("VIT_EPOCHS", "5"))
        batch_size = int(os.getenv("VIT_BATCH", "8"))
        alpha = float(os.getenv("VIT_ALPHA", "0.1"))
        train_vit(images_path, model_path, epochs=epochs, batch_size=batch_size, alpha=alpha)
        raise SystemExit(0)

    if os.getenv("VIT_EVAL", "0") == "1":
        images_path = os.getenv("FLICKR8K_IMAGES_PATH", "data/Flickr8k/Images")
        if not os.path.isdir(images_path):
            raise FileNotFoundError(
                "Images folder not found. Set FLICKR8K_IMAGES_PATH env var or place images in data/Flickr8k/Images."
            )
        model_path = os.getenv("VIT_MODEL_PATH", "vit_stego.pth")
        samples = int(os.getenv("VIT_EVAL_SAMPLES", "512"))
        batch_size = int(os.getenv("VIT_BATCH", "8"))
        alpha = float(os.getenv("VIT_ALPHA", "0.1"))
        evaluate_vit(images_path, model_path, samples=samples, batch_size=batch_size, alpha=alpha)
        raise SystemExit(0)

    decode_image_path = os.getenv("VIT_DECODE_IMAGE")
    if decode_image_path:
        model_path = os.getenv("VIT_MODEL_PATH", "vit_stego.pth")
        output_path = os.getenv("VIT_OUTPUT_PATH", "decoded_secret.png")
        decode_with_vit(model_path, decode_image_path, output_path)
        raise SystemExit(0)

    images_path = os.getenv("FLICKR8K_IMAGES_PATH", "data/Flickr8k/Images")
    if not os.path.isdir(images_path):
        raise FileNotFoundError(
            "Images folder not found. Set FLICKR8K_IMAGES_PATH env var or place images in data/Flickr8k/Images."
        )

    dataset = Flickr8kDataset(images_path, transform=transform)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 images in the dataset folder.")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    cover_image, _ = next(iter(dataloader))
    secret_image, _ = next(iter(dataloader))

    alpha = 0.1
    encoded_image = encode_stegano(cover_image, secret_image, alpha=alpha)
    decoded_secret_image = decode_stegano(encoded_image, cover_image, alpha=alpha)

    show_images(
        [cover_image[1], secret_image[1], encoded_image[1], decoded_secret_image[1]],
        ["Cover", "Secret", "Encoded", "Decoded Secret"],
    )

    try:
        ssim_score, mse_score = calculate_metrics(secret_image[1], decoded_secret_image[1])
        print(f"SSIM: {ssim_score:.4f}")
        print(f"MSE: {mse_score:.4f}")
    except Exception as exc:
        print(f"Metric calculation skipped: {exc}")
