
import math
import os
import random
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
except Exception:
    AutoImageProcessor = None
    AutoModelForSemanticSegmentation = None


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, hue: float = 0.0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, f"images should be a batch of images, but it has shape {images.shape}"
        fn = super().forward
        return torch.stack([fn(img) for img in images], dim=0)


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: Sequence[float]):
        super().__init__(size=size, scale=scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, f"images should be a batch of images, but it has shape {images.shape}"
        fn = super().forward
        return torch.stack([fn(img) for img in images], dim=0)


class BatchNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean.to(images.device)) / self.std.to(images.device)


class OnlineFMIAugment(torch.nn.Module):
    ADE_IDS = [6, 11]

    def __init__(self,
                 water_dir: str | None = None,
                 use_segformer: bool = False,
                 segformer_device: str | None = None,
                 water_level_min: float = 0.50,
                 water_level_max: float = 0.85,
                 alpha_min: float = 0.25,
                 alpha_max: float = 0.55,
                 reflection_strength: float = 0.20,
                 edge_preserve: float = 0.20,
                 wave_strength: float = 0.04,
                 noise_std: float = 0.05,
                 dilate_px: Tuple[int, int] = (5, 11),
                 water_mix_range: Tuple[float, float] = (0.70, 1.00),
                 horizon_range: Tuple[float, float] = (0.35, 0.65)):
        super().__init__()
        self.water_dir = water_dir
        self.use_segformer = use_segformer
        self.segformer_device = segformer_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.water_level_min = water_level_min
        self.water_level_max = water_level_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.reflection_strength = reflection_strength
        self.edge_preserve = edge_preserve
        self.wave_strength = wave_strength
        self.noise_std = noise_std
        self.dilate_px = dilate_px
        self.water_mix_range = water_mix_range
        self.horizon_range = horizon_range
        self._water_textures = self._load_textures(water_dir)

        laplacian = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], dtype=torch.float32)
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, 3, 3))
        self._segformer_processor = None
        self._segformer_model = None
        if self.use_segformer and AutoImageProcessor is not None and AutoModelForSemanticSegmentation is not None:
            try:
                seg_id = "nvidia/segformer-b5-finetuned-ade-640-640"
                self._segformer_processor = AutoImageProcessor.from_pretrained(seg_id)
                self._segformer_model = AutoModelForSemanticSegmentation.from_pretrained(seg_id).to(self.segformer_device).eval()
            except Exception:
                self._segformer_processor = None
                self._segformer_model = None
                self.use_segformer = False

    @staticmethod
    def _load_textures(water_dir: str | None):
        if not water_dir or not os.path.isdir(water_dir):
            return []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return [str(p) for p in Path(water_dir).iterdir() if p.suffix.lower() in exts]

    def _blur(self, x: torch.Tensor, k: int = 7) -> torch.Tensor:
        if k <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        edge = F.conv2d(gray, self.laplacian_kernel.to(x.device), padding=1).abs()
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return edge.repeat(1, 3, 1, 1)

    def _read_random_texture(self, h: int, w: int):
        if not self._water_textures or cv2 is None:
            return None
        tex_path = random.choice(self._water_textures)
        tex = cv2.imread(tex_path, cv2.IMREAD_COLOR)
        if tex is None:
            return None
        sc = random.uniform(0.6, 1.4)
        ang = random.uniform(0, 360)
        tex = cv2.resize(tex, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        M = cv2.getRotationMatrix2D((tex.shape[1] / 2, tex.shape[0] / 2), ang, 1.0)
        tex = cv2.warpAffine(tex, M, (tex.shape[1], tex.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.5:
            tex = cv2.flip(tex, 1)
        if random.random() < 0.5:
            tex = cv2.flip(tex, 0)
        tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC)
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(tex).permute(2, 0, 1).float() / 255.0

    def _segformer_mask(self, rgb: np.ndarray):
        if not self.use_segformer or self._segformer_model is None:
            return None
        try:
            h, w = rgb.shape[:2]
            inp = self._segformer_processor(images=rgb, return_tensors="pt").to(self.segformer_device)
            with torch.no_grad():
                out = self._segformer_model(**inp)
            seg = self._segformer_processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()
            road_mask = np.isin(seg, self.ADE_IDS).astype(np.uint8)
            if road_mask.sum() < 500:
                return None
            if cv2 is not None:
                k = random.randint(*self.dilate_px)
                road_mask = cv2.dilate(road_mask, np.ones((k, k), np.uint8))
            return road_mask
        except Exception:
            return None

    def _fallback_mask(self, h: int, w: int):
        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        mask = (ys >= random.uniform(self.water_level_min, self.water_level_max)).astype(np.float32)
        return np.repeat(mask, w, axis=1)

    def _single(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        pil = TF.to_pil_image(img.detach().cpu().clamp(0, 1))
        rgb = np.array(pil.convert("RGB"))
        h, w = rgb.shape[:2]
        road_mask_np = self._segformer_mask(rgb) if self.use_segformer else None
        if road_mask_np is None:
            road_mask_np = self._fallback_mask(h, w)
        road_mask = torch.from_numpy(road_mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)

        ys = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(1, h, w)
        xs = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(1, h, w)
        waterline_ratio = random.uniform(self.water_level_min, self.water_level_max)
        wave_freq = random.uniform(2.0, 6.0)
        phase = random.uniform(0.0, 2.0 * math.pi)
        waviness = self.wave_strength * torch.sin(2.0 * math.pi * wave_freq * xs + phase)
        surface = waterline_ratio + waviness
        depth_grad = ((ys - surface).clamp(min=0) / max(1e-6, 1.0 - waterline_ratio)).clamp(0, 1)
        alpha = random.uniform(self.alpha_min, self.alpha_max)
        alpha_map = (alpha * (0.35 + 0.65 * depth_grad)) * road_mask
        if self.noise_std > 0:
            alpha_map = (alpha_map + self.noise_std * torch.rand_like(alpha_map)).clamp(0, 1)

        horizon = max(1, int(h * random.uniform(*self.horizon_range)))
        reflection = torch.flip(img[:, :horizon, :], dims=[1])
        reflection = F.interpolate(reflection.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
        reflection = self._blur(reflection.unsqueeze(0), k=9)[0] * road_mask

        texture = self._read_random_texture(h, w)
        if texture is None:
            water_color = torch.tensor([
                random.uniform(0.10, 0.22),
                random.uniform(0.24, 0.40),
                random.uniform(0.28, 0.46),
            ], device=device).view(3, 1, 1).expand_as(img)
            water = water_color
        else:
            texture = texture.to(device)
            mix_w = random.uniform(*self.water_mix_range)
            water = texture * mix_w + reflection * (1.0 - mix_w)
        if random.random() < 0.5:
            tint = torch.tensor([70, 110, 180], device=device, dtype=torch.float32).view(3, 1, 1) / 255.0
            mud = random.uniform(0.25, 0.55)
            water = water * (1.0 - mud) + tint * mud

        darkened = img * (1.0 - 0.18 * road_mask)
        flooded = darkened * (1.0 - alpha_map) + water * alpha_map
        flooded = flooded + self.reflection_strength * reflection * road_mask

        edge_map = self._edge_map(img.unsqueeze(0))[0] * road_mask
        flooded = flooded + self.edge_preserve * edge_map * img
        return flooded.clamp(0.0, 1.0)

    def forward(self, images: torch.Tensor, apply_mask=None) -> torch.Tensor:
        assert images.ndim == 4, f"Expected [B,C,H,W], got {tuple(images.shape)}"
        if apply_mask is None:
            apply_mask = [True] * images.shape[0]
        return torch.stack([self._single(img) if flag else img for img, flag in zip(images, apply_mask)], dim=0)
