"""Face identity preservation reward using face embeddings.

This reward penalizes changes to the subject's face structure (identity)
while allowing texture changes like hair graying.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Literal

from .base import BaseReward


class FaceIdentityReward(BaseReward):
    """Face identity preservation reward using InsightFace buffalo_l.

    Uses face embeddings to detect identity drift. Face embeddings are:
    - Sensitive to geometric structure (face shape, proportions)
    - Robust to texture changes (hair color, skin tone)

    This makes them ideal for detecting the "face changes to a different person"
    failure mode while allowing "hair turns white" which is the desired edit.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        model_size: Literal["large", "small"] = "large",
        detection_threshold: float = 0.5,
        cache_dir: str | None = None,
    ):
        """
        Args:
            device: Device to run the model on
            model_size: Model size ('large' for buffalo_l, 'small' for scratch)
            detection_threshold: Minimum face detection confidence
            cache_dir: Directory to cache model weights
        """
        super().__init__(device)
        self.model_size = model_size
        self.detection_threshold = detection_threshold
        self.cache_dir = cache_dir
        self._source_embeddings: dict[int, torch.Tensor] = {}

    def _load_model(self):
        """Load InsightFace buffalo_l model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "FaceIdentityReward requires insightface. "
                "Install with: pip install insightface"
            )

        # Initialize face analysis model
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if "cuda" in str(self.device) else -1, det_size=(640, 640))
        return app

    def register_source_image(self, image: torch.Tensor) -> None:
        """Register source image for identity comparison.

        Call this once before rollout to capture the source face.

        Args:
            image: Source image [1, 3, H, W] in [0, 1]
        """
        embedding = self._extract_face_embedding(image)
        # Store by hash of image (approximate)
        img_hash = hash(image.cpu().numpy().tobytes()[:1000])
        self._source_embeddings[img_hash] = embedding.detach()

    def get_source_embedding(self, image: torch.Tensor) -> torch.Tensor | None:
        """Get stored source embedding for an image."""
        img_hash = hash(image.cpu().numpy().tobytes()[:1000])
        return self._source_embeddings.get(img_hash)

    def _extract_face_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Extract face embedding from an image.

        Args:
            image: Image tensor [B, 3, H, W] in [0, 1]

        Returns:
            Face embedding tensor [B, embedding_dim]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise ValueError("FaceIdentityReward processes one image at a time")

        # Convert to numpy for insightface
        img_np = (
            image.detach()
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        # InsightFace expects BGR [0, 255]
        img_np = (img_np * 255).astype("uint8")
        img_np = img_np[:, :, ::-1]  # RGB to BGR

        # Detect and extract face
        faces = self.model.get(img_np)

        if len(faces) == 0:
            # No face detected - return zero embedding
            return torch.zeros(512, device=image.device, dtype=image.dtype)

        # Use the largest face
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding = torch.from_numpy(face.embedding).float().to(image.device)

        # Normalize
        embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
        return embedding

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute face identity similarity reward.

        Args:
            image: Edited image [B, 3, H, W] in [0, 1]
            prompt: Text prompt (not used for face identity)

        Returns:
            Scalar reward (higher = more similar to source identity)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        source_embedding = self.get_source_embedding(image)
        if source_embedding is None:
            # No source registered - register this as source
            self.register_source_image(image)
            return torch.tensor(1.0, device=image.device, dtype=image.dtype)

        # Extract embedding from current image
        edited_embedding = self._extract_face_embedding(image)

        # Compute cosine similarity
        similarity = F.cosine_similarity(
            edited_embedding.unsqueeze(0),
            source_embedding.unsqueeze(0),
            dim=1,
        )

        # Return as scalar reward
        return similarity.mean()

    def compute_penalty(
        self,
        image: torch.Tensor,
        source_image: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """Compute identity penalty for gradient-based optimization.

        This returns a PENALTY (lower is better), suitable for
        gradient descent on the loss.

        Args:
            image: Edited image [1, 3, H, W]
            source_image: Source image [1, 3, H, W]
            intensity: Scaling factor for penalty

        Returns:
            Penalty tensor (lower = better identity preservation)
        """
        # Register source if not already
        if self.get_source_embedding(source_image) is None:
            self.register_source_image(source_image)

        # Compute reward
        reward = self(source_image, "")  # Get source embedding registered
        reward = self(image, "")  # Get edited embedding

        # Get embeddings
        src_emb = self.get_source_embedding(source_image)
        edit_emb = self._extract_face_embedding(image)

        # Cosine distance (lower = more similar)
        distance = 1.0 - F.cosine_similarity(
            edit_emb.unsqueeze(0), src_emb.unsqueeze(0), dim=1
        )

        return intensity * distance


class FaceDetectionReward(BaseReward):
    """Simpler face-based reward using MTCNN + ArcFace.

    This is a fallback if insightface is not available.
    Uses a simpler approach with face detection + embedding.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        cache_dir: str | None = None,
    ):
        super().__init__(device)
        self.cache_dir = cache_dir

    def _load_model(self):
        """Load face detection and recognition models."""
        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError:
            raise ImportError(
                "FaceDetectionReward requires facenet_pytorch. "
                "Install with: pip install facenet-pytorch"
            )

        # MTCNN for detection
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            device=self.device,
            post_process=False,
        )

        # InceptionResnetV1 for embedding
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        return {"mtcnn": mtcnn, "resnet": resnet}

    def _extract_face(self, image: torch.Tensor) -> torch.Tensor | None:
        """Extract face embedding from image."""
        if image.shape[-2:] != (160, 160):
            # Resize to required size
            image = F.interpolate(image, size=(160, 160), mode="bicubic")

        # Get face crops
        faces = self.model["mtcnn"](image)

        if faces is None or len(faces) == 0:
            return None

        # Use largest detected face
        face = faces[0] if faces.dim() == 4 else faces

        # Get embedding
        with torch.no_grad():
            embedding = self.model["resnet"](face.unsqueeze(0))

        return embedding.squeeze(0)

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute face presence reward."""
        embedding = self._extract_face(image)

        if embedding is None:
            return torch.tensor(0.0, device=image.device)

        # Higher embedding norm = more face-like
        return embedding.norm(p=2, dim=-1).mean()
