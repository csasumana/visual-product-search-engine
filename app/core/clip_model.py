from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image

from app.core.config import settings


class CLIPService:
    def __init__(self):
        self.device = self._resolve_device(settings.DEVICE)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.CLIP_MODEL_NAME,
            pretrained=settings.CLIP_PRETRAINED,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)
        self.model.eval()

    def _resolve_device(self, configured_device: str) -> str:
        if configured_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_tensor)
        features = features.cpu().numpy().astype(np.float32)
        features = self._l2_normalize(features)
        return features

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features.cpu().numpy().astype(np.float32)
        features = self._l2_normalize(features)
        return features

    @torch.no_grad()
    def embed_image_batch(self, images: List[Image.Image]) -> np.ndarray:
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(image_tensors)
        features = features.cpu().numpy().astype(np.float32)
        features = self._l2_normalize(features)
        return features

    @property
    def embedding_dim(self) -> int:
        dummy = self.embed_text("test")
        return int(dummy.shape[1])