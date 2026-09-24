"""Reward models for VeloEdit trajectory optimization.

This module provides modular reward functions for detecting and correcting:
1. Identity drift (face embedding based)
2. Spatial layout drift (DINOv2 based)
3. Semantic alignment (SigLIP based)
4. Perceptual quality (DreamSim/LPIPS based)
"""

from .base import RewardFn, RewardOutput
from .face_identity import FaceIdentityReward, FaceDetectionReward
from .spatial_layout import SpatialLayoutReward
from .semantic import SemanticReward, CLIPSemanticReward
from .perception import DreamSimReward, LPIPSReward
from .composite import CompositeReward, RewardWeights, CompositeRewardConfig, IntensityAwareReward

__all__ = [
    # Base
    "RewardFn",
    "RewardOutput",
    # Identity
    "FaceIdentityReward",
    "FaceDetectionReward",
    # Layout
    "SpatialLayoutReward",
    # Semantic
    "SemanticReward",
    "CLIPSemanticReward",
    # Perception
    "DreamSimReward",
    "LPIPSReward",
    # Composite
    "CompositeReward",
    "RewardWeights",
    "CompositeRewardConfig",
    "IntensityAwareReward",
]
