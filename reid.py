# -*- coding: utf-8 -*-
"""
Person Re-ID embeddings via OSNet (torchreid).

Uses ImageNet-pretrained OSNet x1.0 (512-D features). In eval mode the backbone
returns L2-friendly feature vectors before the classification head.

Requires: pip install torchreid gdown
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

REID_EMBEDDING_DIM = 512

if TYPE_CHECKING:
    pass


class OSNetReIDExtractor:
    """Batch Re-ID feature extraction for person crops (PIL RGB)."""

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        image_size: tuple[int, int] = (256, 128),
        verbose: bool = False,
    ):
        try:
            from torchreid.reid.models import build_model
        except ImportError as e:
            raise ImportError(
                "torchreid is required for person Re-ID. Install with: pip install torchreid gdown"
            ) from e

        self.device = torch.device(device)
        # torchreid only understands CUDA for use_gpu flag; MPS/CPU use CPU build then .to()
        use_gpu = device.startswith("cuda")
        self.model = build_model(
            model_name,
            num_classes=1,
            loss="softmax",
            pretrained=True,
            use_gpu=use_gpu,
        )
        self.model.eval()
        self.model.to(self.device)

        self._preprocess = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if verbose:
            nparams = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"   OSNet Re-ID loaded ({model_name}), ~{nparams:.2f}M params, device={self.device}")

    def _pil_batch_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self._preprocess(img.convert("RGB")) for img in images]
        return torch.stack(tensors, dim=0).to(self.device)

    @torch.inference_mode()
    def _forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Returns raw features [B, 512] (eval mode skips classifier)."""
        return self.model(batch)

    def generate_embeddings_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """L2-normalized embeddings, shape (N, 512)."""
        if not images:
            return np.zeros((0, REID_EMBEDDING_DIM), dtype=np.float32)

        all_feats: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            t = self._pil_batch_to_tensor(chunk)
            feats = self._forward_batch(t)
            all_feats.append(feats)

        stacked = torch.cat(all_feats, dim=0)
        stacked = F.normalize(stacked, p=2, dim=1)
        return stacked.cpu().numpy().astype(np.float32)

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        return self.generate_embeddings_batch([image])[0]

    def get_embedding_dim(self) -> int:
        return REID_EMBEDDING_DIM


def create_reid_extractor(device: str, model_name: str = "osnet_x1_0", verbose: bool = False) -> OSNetReIDExtractor:
    """Factory: maps logical device string to OSNet loader."""
    return OSNetReIDExtractor(model_name=model_name, device=device, verbose=verbose)
