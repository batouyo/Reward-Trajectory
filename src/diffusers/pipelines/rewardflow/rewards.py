from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import PIL
import torch
import torch.nn.functional as F


def _build_hf_kwargs(
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
    token: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    device_map: str | dict | None = None,
) -> dict:
    hf_kwargs = {}
    if cache_dir is not None:
        hf_kwargs["cache_dir"] = cache_dir
    if local_files_only is not None:
        hf_kwargs["local_files_only"] = local_files_only
    if token is not None:
        hf_kwargs["token"] = token
    if revision is not None:
        hf_kwargs["revision"] = revision
    if trust_remote_code is not None:
        hf_kwargs["trust_remote_code"] = trust_remote_code
    if device_map is not None:
        hf_kwargs["device_map"] = device_map
    return hf_kwargs


def _extract_features(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]

    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if value is None:
                continue
            if attr == "last_hidden_state" and value.dim() >= 3:
                return value.mean(dim=1)
            return value

    raise TypeError(f"Unsupported model output type for features: {type(output)}")


def _get_vision_attr(cfg: object, name: str, fallback=None):
    if hasattr(cfg, name):
        return getattr(cfg, name)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None and hasattr(vision_cfg, name):
        return getattr(vision_cfg, name)
    return fallback


class RewardFn(Protocol):
    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Return a scalar reward (higher is better)."""


def _reward_device(reward: object, fallback: torch.device) -> torch.device:
    """Resolve a reward's current device after any onload operation."""

    model = getattr(reward, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    explicit_device = getattr(reward, "device", None)
    if explicit_device is not None:
        try:
            return torch.device(explicit_device)
        except (TypeError, RuntimeError):
            pass
    return fallback


@dataclass
class RewardGuidanceConfig:
    temperature: float = 1.0
    ema_decay: float = 0.99
    clip_value: float = 5.0
    weight_floor: float = 0.0
    eps: float = 1e-7
    change: float = 1e-50


class RewardGuidance:
    def __init__(
        self,
        rewards: Sequence[RewardFn],
        temperature: float = 1.0,
        ema_decay: float = 0.99,
        clip_value: float = 5.0,
        weight_floor: float = 0.0,
        eps: float = 1e-6,
    ):
        if not rewards:
            raise ValueError("RewardGuidance requires at least one reward function.")
        self.rewards = list(rewards)
        self.temperature = float(temperature)
        self.ema_decay = float(ema_decay)
        self.clip_value = float(clip_value)
        self.weight_floor = float(weight_floor)
        self.eps = float(eps)
        self._ema_mean = None
        self._ema_var = None

    def _update_ema(self, values: torch.Tensor):
        if self._ema_mean is None:
            self._ema_mean = values.detach()
            self._ema_var = torch.zeros_like(values.detach())
            return

        decay = self.ema_decay
        mean = self._ema_mean
        var = self._ema_var
        mean = decay * mean + (1 - decay) * values.detach()
        var = decay * var + (1 - decay) * (values.detach() - mean) ** 2
        self._ema_mean = mean
        self._ema_var = var

    def compute(self, image: torch.Tensor, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _reward_device(reward: object, fallback: torch.device) -> torch.device:
            if hasattr(reward, "model"):
                try:
                    return next(reward.model.parameters()).device
                except StopIteration:
                    return fallback
            if hasattr(reward, "device"):
                try:
                    return torch.device(reward.device)
                except Exception:
                    return fallback
            return fallback

        reward_values = []
        base_device = image.device
        for reward in self.rewards:
            maybe_onload = getattr(reward, "maybe_onload", None)
            if callable(maybe_onload):
                maybe_onload()
            target_device = _reward_device(reward, base_device)
            reward_image = image if target_device == base_device else image.to(target_device)
            value = reward(image=reward_image, prompt=prompt)
            if value.dim() > 0:
                value = value.mean()
            if value.device != base_device:
                value = value.to(base_device)
            reward_values.append(value)

        reward_values = torch.stack(reward_values)
        self._update_ema(reward_values)

        mean = self._ema_mean.to(reward_values.device)
        var = self._ema_var.to(reward_values.device)
        normalized = (reward_values - mean) / torch.sqrt(var + self.eps)
        if self.clip_value > 0:
            normalized = torch.clamp(normalized, -self.clip_value, self.clip_value)

        temperature = max(self.temperature, 1e-6)
        weights = torch.softmax(normalized / temperature, dim=0)
        if self.weight_floor > 0:
            weights = weights + self.weight_floor
            weights = weights / weights.sum()

        total_reward = (weights * reward_values).sum()
        return total_reward, reward_values, weights

    def maybe_offload(self):
        for reward in self.rewards:
            maybe_offload = getattr(reward, "maybe_offload", None)
            if callable(maybe_offload):
                maybe_offload()


class ResearchStaticRewardGuidance:
    """Research utility for transparent fusion with caller-supplied weights.

    This is not a reproduction or static approximation of RewardFlow's
    prompt-aware adaptive softmax policy. Signed weights are intentionally
    supported for controlled research compositions.

    Unlike the legacy ``RewardGuidance``, this class performs no running
    normalization, softmax, scheduling, or value-dependent reweighting.
    """

    def __init__(self, rewards: dict[str, RewardFn], weights: dict[str, float]):
        if not rewards:
            raise ValueError("StaticRewardGuidance requires at least one named reward.")
        if set(rewards) != set(weights):
            missing = sorted(set(rewards) - set(weights))
            extra = sorted(set(weights) - set(rewards))
            raise ValueError(f"Reward and weight names must match exactly; missing={missing}, extra={extra}.")
        if any(not math.isfinite(float(weight)) for weight in weights.values()):
            raise ValueError("Static reward weights must be finite.")
        self.rewards = dict(rewards)
        self.weights = {name: float(weight) for name, weight in weights.items()}

    def compute(self, image: torch.Tensor, prompt: str | list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = {}
        total = None
        for name, reward in self.rewards.items():
            maybe_onload = getattr(reward, "maybe_onload", None)
            if callable(maybe_onload):
                maybe_onload()
            target_device = _reward_device(reward, image.device)
            # Device copies are autograd operations. Do not detach here: the
            # reward gradient must cross back to the original clean image.
            reward_image = image if target_device == image.device else image.to(target_device)
            value = reward(image=reward_image, prompt=prompt)
            if not torch.is_tensor(value):
                raise TypeError(f"Reward `{name}` must return a torch.Tensor, got {type(value)}.")
            if value.dim() > 0:
                value = value.mean()
            if value.device != image.device:
                value = value.to(image.device)
            values[name] = value
            weighted = self.weights[name] * value
            total = weighted if total is None else total + weighted
        return total, values

    def maybe_offload(self):
        for reward in self.rewards.values():
            maybe_offload = getattr(reward, "maybe_offload", None)
            if callable(maybe_offload):
                maybe_offload()


# Backwards-compatible name for the first paper-mode implementation. New code
# should prefer the explicit research-oriented name above.
StaticRewardGuidance = ResearchStaticRewardGuidance


class SigLIPReward:
    """
    Differentiable text-image similarity reward using SigLIP.

    This avoids PIL-based preprocess to keep gradients flowing by using
    torch resizing + normalization. Customize `image_size`, `image_mean`,
    and `image_std` to match the checkpoint.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        image_size: int | None = None,
        image_mean: tuple[float, float, float] | None = None,
        image_std: tuple[float, float, float] | None = None,
    ):
        try:
            from transformers import SiglipModel, SiglipTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("SigLIPReward requires transformers with SiglipModel.") from exc

        hf_kwargs = _build_hf_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        hf_model_kwargs = dict(hf_kwargs)
        if dtype is not None:
            hf_model_kwargs.setdefault("torch_dtype", dtype)

        self.model = SiglipModel.from_pretrained(model_id, **hf_model_kwargs)
        self.tokenizer = SiglipTokenizer.from_pretrained(model_id, **hf_kwargs)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

        cfg = getattr(self.model, "config", None)
        self.image_size = int(image_size or _get_vision_attr(cfg, "image_size", 224))
        self.image_mean = image_mean or tuple(_get_vision_attr(cfg, "image_mean", (0.5, 0.5, 0.5)))
        self.image_std = image_std or tuple(_get_vision_attr(cfg, "image_std", (0.5, 0.5, 0.5)))
        self._cached_prompt = None
        self._cached_text_features = None

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        self._cached_prompt = prompt
        self._cached_text_features = self._encode_text(prompt, device=device)

    def _encode_text(self, prompt: str | list[str], device: torch.device | None = None) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = _extract_features(self.model.get_text_features(**inputs))
        return F.normalize(text_features, dim=-1)

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            image = image.float()
        image = torch.clamp(image, 0, 1)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        mean = torch.tensor(self.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        device = image.device
        if self._cached_text_features is None or self._cached_prompt != prompt:
            text_features = self._encode_text(prompt, device=device)
        else:
            text_features = self._cached_text_features
            if text_features.device != device:
                text_features = text_features.to(device)

        image_inputs = self._preprocess_image(image)
        image_features = _extract_features(self.model.get_image_features(pixel_values=image_inputs))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (image_features * text_features).sum(dim=-1)
        return similarity.mean()


class RegionCLIPReward:
    """
    Placeholder for RegionCLIP-style rewards.

    Default implementation uses CLIP global similarity. Swap in a region-aware
    model or override `__call__` for region-level scoring.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        image_size: int | None = None,
        image_mean: tuple[float, float, float] | None = None,
        image_std: tuple[float, float, float] | None = None,
    ):
        try:
            from transformers import CLIPModel, CLIPTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("RegionCLIPReward requires transformers with CLIPModel.") from exc

        hf_kwargs = _build_hf_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        hf_model_kwargs = dict(hf_kwargs)
        if dtype is not None:
            hf_model_kwargs.setdefault("torch_dtype", dtype)

        self.model = CLIPModel.from_pretrained(model_id, **hf_model_kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, **hf_kwargs)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

        cfg = getattr(self.model, "config", None)
        self.image_size = int(image_size or _get_vision_attr(cfg, "image_size", 224))
        self.image_mean = image_mean or (0.48145466, 0.4578275, 0.40821073)
        self.image_std = image_std or (0.26862954, 0.26130258, 0.27577711)
        self._cached_prompt = None
        self._cached_text_features = None

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        self._cached_prompt = prompt
        self._cached_text_features = self._encode_text(prompt, device=device)

    def _encode_text(self, prompt: str | list[str], device: torch.device | None = None) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = _extract_features(self.model.get_text_features(**inputs))
        return F.normalize(text_features, dim=-1)

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            image = image.float()
        image = torch.clamp(image, 0, 1)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        mean = torch.tensor(self.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        device = image.device
        if self._cached_text_features is None or self._cached_prompt != prompt:
            text_features = self._encode_text(prompt, device=device)
        else:
            text_features = self._cached_text_features
            if text_features.device != device:
                text_features = text_features.to(device)

        image_inputs = self._preprocess_image(image)
        image_features = _extract_features(self.model.get_image_features(pixel_values=image_inputs))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (image_features * text_features).sum(dim=-1)
        return similarity.mean()


class LLMReward:
    """
    Generic wrapper for an external reward function (e.g., VLM/LLM scoring).
    The callable must accept (image, prompt) and return a scalar tensor.
    """

    def __init__(self, score_fn):
        if score_fn is None:
            raise ValueError("LLMReward requires a callable score_fn.")
        self.score_fn = score_fn

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        if hasattr(self.score_fn, "prepare"):
            self.score_fn.prepare(prompt=prompt, device=device)

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        return self.score_fn(image=image, prompt=prompt)


def qwen_vqa_token_reward(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    margin: float,
    lambda_margin: float,
) -> torch.Tensor:
    """Compute the teacher-forced VQA token reward from aligned next-token logits.

    ASSUMPTION: Eq. 4 is typeset ambiguously. We implement the prose description
    "negative cross-entropy plus margin objective" so larger reward means larger
    target log-likelihood and a larger target-vs-best-other logit margin.
    """

    if logits.ndim != 2:
        raise ValueError("`logits` must have shape [answer_tokens, vocabulary].")
    if target_token_ids.ndim != 1 or target_token_ids.shape[0] != logits.shape[0]:
        raise ValueError("`target_token_ids` must align one-to-one with `logits` rows.")
    if margin < 0 or lambda_margin < 0:
        raise ValueError("`margin` and `lambda_margin` must be non-negative.")

    logits = logits.float()
    target_token_ids = target_token_ids.to(device=logits.device, dtype=torch.long)
    correct_logits = logits.gather(1, target_token_ids[:, None]).squeeze(1)
    log_prob_correct = F.log_softmax(logits, dim=-1).gather(1, target_token_ids[:, None]).squeeze(1)

    top_values, top_indices = logits.topk(k=2, dim=-1)
    max_other = torch.where(top_indices[:, 0] == target_token_ids, top_values[:, 1], top_values[:, 0])
    margin_penalty = F.relu(float(margin) - correct_logits + max_other)
    return (log_prob_correct - float(lambda_margin) * margin_penalty).mean()


class Qwen25VQAReward:
    """Differentiable teacher-forced VQA reward using frozen Qwen2.5-VL 3B.

    Images remain torch tensors throughout preprocessing. The current adapter
    supports one image because RewardFlow's paper experiments use batch size one
    for editing and the paper does not specify multi-image Q&A association.
    """

    def __init__(
        self,
        question: str,
        answer: str,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        margin: float | None = None,
        lambda_margin: float | None = None,
        max_answer_tokens: int = 70,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = True,
    ):
        # ASSUMPTION: The paper does not disclose experimental values for the
        # token margin or its coefficient, so both must be caller-supplied.
        if not question.strip() or not answer.strip():
            raise ValueError("Qwen25VQAReward requires non-empty `question` and `answer`.")
        if margin is None or lambda_margin is None:
            raise ValueError(
                "Qwen25VQAReward requires explicit `margin` and `lambda_margin`; the paper values are unavailable."
            )
        if margin < 0 or lambda_margin < 0:
            raise ValueError("`margin` and `lambda_margin` must be non-negative.")
        if not 1 <= max_answer_tokens <= 70:
            raise ValueError("`max_answer_tokens` must be between 1 and the paper's approximate cap of 70.")

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Qwen25VQAReward requires Qwen2.5-VL support in transformers.") from exc

        hf_kwargs = _build_hf_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs = dict(hf_kwargs)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.processor = AutoProcessor.from_pretrained(model_id, **hf_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.question = question.strip()
        self.answer = answer.strip()
        self.margin = float(margin)
        self.lambda_margin = float(lambda_margin)
        self.max_answer_tokens = int(max_answer_tokens)
        self._validate_image_processor()

    def _validate_image_processor(self) -> None:
        required = (
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "min_pixels",
            "max_pixels",
            "image_mean",
            "image_std",
        )
        missing = [name for name in required if not hasattr(self.processor.image_processor, name)]
        if missing:
            raise NotImplementedError(
                "Cannot reproduce differentiable Qwen image preprocessing; processor is missing " + ", ".join(missing)
            )

    @staticmethod
    def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Qwen2.5-VL requires an absolute image aspect ratio below 200.")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return resized_height, resized_width

    def _differentiable_image_inputs(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ASSUMPTION: The paper reports batch-one editing and does not define
        # how Q&A pairs map to batched or multi-image reward inputs.
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise NotImplementedError("Qwen25VQAReward currently supports exactly one RGB image tensor [1, 3, H, W].")
        if not image.is_floating_point():
            raise TypeError("Qwen25VQAReward expects a floating-point image tensor in [0, 1].")

        processor = self.processor.image_processor
        patch_size = int(processor.patch_size)
        temporal_patch_size = int(processor.temporal_patch_size)
        merge_size = int(processor.merge_size)
        factor = patch_size * merge_size
        height, width = image.shape[-2:]
        resized_height, resized_width = self._smart_resize(
            height,
            width,
            factor,
            int(processor.min_pixels),
            int(processor.max_pixels),
        )
        # ASSUMPTION: Hugging Face's reference processor uses PIL/NumPy bicubic
        # resizing, which destroys gradients. Torch bicubic with antialiasing is
        # the closest differentiable operation, but is not bit-identical to PIL.
        image = F.interpolate(
            image.clamp(0, 1),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(processor.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std

        frames = image.repeat(temporal_patch_size, 1, 1, 1)
        channel = frames.shape[1]
        grid_t = frames.shape[0] // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        patches = frames.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        pixel_values = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=image.device, dtype=torch.long)
        return pixel_values, image_grid_thw

    def _teacher_forced_inputs(self, image_grid_thw: torch.Tensor, device: torch.device):
        # ASSUMPTION: The paper does not define the exact Qwen chat template or
        # whether the assistant terminator belongs to a*. We use the checkpoint's
        # official chat template as prefix and score answer tokens only.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_token = self.processor.image_token
        if prompt_text.count(image_token) != 1:
            raise NotImplementedError("Expected exactly one Qwen image placeholder in the chat template.")
        merge_length = int(self.processor.image_processor.merge_size) ** 2
        image_token_count = int(image_grid_thw[0].prod().item() // merge_length)
        prompt_text = prompt_text.replace(image_token, image_token * image_token_count, 1)

        tokenizer = self.processor.tokenizer
        prefix_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        answer_ids = tokenizer(self.answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
        answer_ids = answer_ids[: self.max_answer_tokens]
        if answer_ids.numel() == 0:
            raise ValueError("The target answer produced no tokens.")
        input_ids = torch.cat([prefix_ids, answer_ids]).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        target_positions = torch.arange(
            prefix_ids.numel() - 1,
            prefix_ids.numel() + answer_ids.numel() - 1,
            device=device,
        )
        return input_ids, attention_mask, answer_ids.to(device), target_positions

    def _model_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        try:
            model_device = next(self.model.parameters()).device
            visual = getattr(self.model, "visual", None)
            if visual is None:
                visual = self.model.model.visual
            return model_device, visual.dtype
        except (StopIteration, AttributeError) as exc:
            raise NotImplementedError(
                "Cannot determine Qwen2.5-VL model device/dtype for differentiable input."
            ) from exc

    def _aligned_answer_logits(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen and return next-token logits aligned to the target answer."""

        model_device, model_dtype = self._model_device_and_dtype()
        pixel_values = pixel_values.to(device=model_device, dtype=model_dtype)
        image_grid_thw = image_grid_thw.to(model_device)
        input_ids, attention_mask, answer_ids, target_positions = self._teacher_forced_inputs(
            image_grid_thw, model_device
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[0, target_positions], answer_ids

    @staticmethod
    def _fidelity_image_pair(image: PIL.Image.Image | torch.Tensor) -> tuple[PIL.Image.Image, torch.Tensor]:
        """Create quantization-aligned PIL and [1, C, H, W] tensor views for diagnostics."""

        if isinstance(image, PIL.Image.Image):
            pil_image = image.convert("RGB")
        elif torch.is_tensor(image):
            tensor = image.detach().to(device="cpu", dtype=torch.float32)
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError("Processor fidelity diagnostics support one image.")
                tensor = tensor[0]
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError("Processor fidelity diagnostics require RGB tensor shape [3, H, W].")
            array = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            pil_image = PIL.Image.fromarray(array, mode="RGB")
        else:
            raise TypeError("Processor fidelity diagnostics require a PIL image or torch tensor.")

        array = np.asarray(pil_image, dtype=np.uint8).copy()
        tensor_image = torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)
        return pil_image, tensor_image

    def compare_with_official_processor(self, image: PIL.Image.Image | torch.Tensor) -> dict[str, object]:
        """Compare the differentiable adapter with the official Qwen processor.

        This test/debug helper intentionally uses a non-differentiable PIL
        reference path. Production reward evaluation never calls it.
        """

        pil_image, tensor_image = self._fidelity_image_pair(image)
        official = self.processor.image_processor(images=pil_image, return_tensors="pt")
        official_pixels = official["pixel_values"].float()
        official_grid = official["image_grid_thw"].to(torch.long)
        differentiable_pixels, differentiable_grid = self._differentiable_image_inputs(tensor_image)
        differentiable_pixels = differentiable_pixels.float()

        pixel_difference = differentiable_pixels - official_pixels
        pixel_cosine = F.cosine_similarity(
            differentiable_pixels.reshape(1, -1), official_pixels.reshape(1, -1), dim=1
        )[0]

        # This helper is diagnostic only. The production differentiable path
        # remains grad-enabled through `_aligned_answer_logits` and `__call__`.
        with torch.no_grad():
            official_logits, official_targets = self._aligned_answer_logits(official_pixels, official_grid)
            differentiable_logits, differentiable_targets = self._aligned_answer_logits(
                differentiable_pixels, differentiable_grid
            )
            if not torch.equal(official_targets, differentiable_targets):
                raise RuntimeError("Official and differentiable paths produced different answer token IDs.")

            official_logits = official_logits.float()
            differentiable_logits = differentiable_logits.float()
            target_ids = official_targets[:, None]
            official_target_logits = official_logits.gather(1, target_ids).squeeze(1)
            differentiable_target_logits = differentiable_logits.gather(1, target_ids).squeeze(1)
            official_logprobs = F.log_softmax(official_logits, dim=-1).gather(1, target_ids).squeeze(1)
            differentiable_logprobs = F.log_softmax(differentiable_logits, dim=-1).gather(1, target_ids).squeeze(1)
            official_reward = qwen_vqa_token_reward(
                official_logits,
                official_targets,
                margin=self.margin,
                lambda_margin=self.lambda_margin,
            )
            differentiable_reward = qwen_vqa_token_reward(
                differentiable_logits,
                differentiable_targets,
                margin=self.margin,
                lambda_margin=self.lambda_margin,
            )

        reward_difference = differentiable_reward - official_reward
        return {
            "official_image_grid_thw": official_grid.tolist(),
            "differentiable_image_grid_thw": differentiable_grid.cpu().tolist(),
            "official_pixel_values_shape": list(official_pixels.shape),
            "differentiable_pixel_values_shape": list(differentiable_pixels.shape),
            "pixel_mse": float(pixel_difference.square().mean()),
            "pixel_mae": float(pixel_difference.abs().mean()),
            "pixel_cosine": float(pixel_cosine),
            "pixel_max_abs_difference": float(pixel_difference.abs().max()),
            "aligned_logits_mse": float((differentiable_logits - official_logits).square().mean()),
            "target_logits_mse": float((differentiable_target_logits - official_target_logits).square().mean()),
            "target_token_logprob_mean_difference": float((differentiable_logprobs - official_logprobs).mean()),
            "target_token_logprob_mae": float((differentiable_logprobs - official_logprobs).abs().mean()),
            "official_reward": float(official_reward),
            "differentiable_reward": float(differentiable_reward),
            "reward_difference": float(reward_difference),
            "reward_abs_difference": float(reward_difference.abs()),
        }

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        return None

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        pixel_values, image_grid_thw = self._differentiable_image_inputs(image)
        aligned_logits, answer_ids = self._aligned_answer_logits(pixel_values, image_grid_thw)
        return qwen_vqa_token_reward(
            aligned_logits,
            answer_ids,
            margin=self.margin,
            lambda_margin=self.lambda_margin,
        ).to(image.device)


# class Qwen3VLCaptionReward:
#     """
#     Qwen3-VL reward: generate a caption from the image, then compute the
#     image-conditioned LM loss on that caption. Reward = -loss.
#     """

#     def __init__(
#         self,
#         model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#         device_map: str | dict | None = None,
#         quantization_config: object | None = None,
#         load_in_4bit: bool | None = None,
#         load_in_8bit: bool | None = None,
#         bnb_4bit_compute_dtype: torch.dtype | None = None,
#         bnb_4bit_quant_type: str = "nf4",
#         bnb_4bit_use_double_quant: bool = True,
#         offload_to_cpu: bool = False,
#         onload_device: torch.device | str | None = None,
#         cache_dir: str | None = None,
#         local_files_only: bool | None = None,
#         token: str | None = None,
#         revision: str | None = None,
#         caption_prompt: str = "Describe the image in detail.",
#         max_new_tokens: int = 64,
#         do_sample: bool = False,
#         temperature: float = 1.0,
#         trust_remote_code: bool = True,
#     ):
#         try:
#             from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
#         except Exception as exc:  # pragma: no cover - optional dependency
#             raise ImportError("Qwen3VLCaptionReward requires transformers.") from exc

#         hf_kwargs = _build_hf_kwargs(
#             cache_dir=cache_dir,
#             local_files_only=local_files_only,
#             token=token,
#             revision=revision,
#             trust_remote_code=trust_remote_code,
#             device_map=device_map,
#         )
#         hf_model_kwargs = dict(hf_kwargs)
#         if dtype is not None:
#             hf_model_kwargs.setdefault("torch_dtype", dtype)

#         is_quantized = False
#         if quantization_config is not None:
#             hf_model_kwargs["quantization_config"] = quantization_config
#             is_quantized = True
#         elif load_in_4bit or load_in_8bit:
#             try:
#                 from transformers import BitsAndBytesConfig
#             except Exception as exc:  # pragma: no cover - optional dependency
#                 raise ImportError("Qwen3VLCaptionReward quantization requires bitsandbytes + transformers.") from exc

#             compute_dtype = bnb_4bit_compute_dtype if bnb_4bit_compute_dtype is not None else dtype
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=bool(load_in_4bit),
#                 load_in_8bit=bool(load_in_8bit),
#                 bnb_4bit_compute_dtype=compute_dtype,
#                 bnb_4bit_quant_type=bnb_4bit_quant_type,
#                 bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
#             )
#             hf_model_kwargs["quantization_config"] = bnb_config
#             is_quantized = True

#         if is_quantized and device_map is None and device is not None:
#             hf_model_kwargs["device_map"] = {"": device}

#         self.processor = AutoProcessor.from_pretrained(model_id, **hf_kwargs)
#         try:
#             self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **hf_model_kwargs)
#         except Exception:
#             self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **hf_model_kwargs)

#         self.offload_to_cpu = bool(offload_to_cpu)
#         self.onload_device = torch.device(onload_device) if onload_device is not None else None
#         self._can_move_model = (
#             not hasattr(self.model, "hf_device_map")
#             and not getattr(self.model, "is_loaded_in_4bit", False)
#             and not getattr(self.model, "is_loaded_in_8bit", False)
#         )

#         if not is_quantized:
#             if device is not None:
#                 self.model = self.model.to(device)
#             if dtype is not None:
#                 self.model = self.model.to(dtype=dtype)
#         self.model.eval()

#         self.caption_prompt = caption_prompt
#         self.max_new_tokens = int(max_new_tokens)
#         self.do_sample = bool(do_sample)
#         self.temperature = float(temperature)
#         self._cached_prompt = None
#         self._cached_prompt_text = None

#     def prepare(self, prompt: str | list[str], device: torch.device | None = None):
#         self._cached_prompt = prompt
#         self._cached_prompt_text = None

#     def maybe_onload(self):
#         if not self.offload_to_cpu:
#             return
#         if not self._can_move_model:
#             return
#         target = self.onload_device
#         if target is None:
#             return
#         try:
#             self.model.to(target)
#         except Exception as exc:
#             warnings.warn(f"Qwen3VLCaptionReward failed to onload to {target}: {exc}")

#     def maybe_offload(self):
#         if not self.offload_to_cpu:
#             return
#         if not self._can_move_model:
#             return
#         try:
#             self.model.to("cpu")
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         except Exception as exc:
#             warnings.warn(f"Qwen3VLCaptionReward failed to offload to CPU: {exc}")


class Qwen25VLCaptionReward:
    """
    Qwen2.5-VL reward: generate a caption from the image, then compute the
    image-conditioned LM loss on that caption. Reward = -loss.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        device_map: str | dict | None = None,
        quantization_config: object | None = None,
        load_in_4bit: bool | None = None,
        load_in_8bit: bool | None = None,
        bnb_4bit_compute_dtype: torch.dtype | None = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        offload_to_cpu: bool = False,
        onload_device: torch.device | str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        caption_prompt: str = "Describe the image in detail.",
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        trust_remote_code: bool = True,
    ):
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Qwen25VLCaptionReward requires Qwen2.5-VL + qwen_vl_utils.") from exc

        self._process_vision_info = process_vision_info

        hf_kwargs = _build_hf_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
        hf_model_kwargs = dict(hf_kwargs)
        if dtype is not None:
            hf_model_kwargs.setdefault("torch_dtype", dtype)

        is_quantized = False
        if quantization_config is not None:
            hf_model_kwargs["quantization_config"] = quantization_config
            is_quantized = True
        elif load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("Qwen25VLCaptionReward quantization requires bitsandbytes + transformers.") from exc

            compute_dtype = bnb_4bit_compute_dtype if bnb_4bit_compute_dtype is not None else dtype
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=bool(load_in_4bit),
                load_in_8bit=bool(load_in_8bit),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
            hf_model_kwargs["quantization_config"] = bnb_config
            is_quantized = True

        if is_quantized and device_map is None and device is not None:
            hf_model_kwargs["device_map"] = {"": device}

        self.processor = AutoProcessor.from_pretrained(model_id, **hf_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **hf_model_kwargs)

        self.offload_to_cpu = bool(offload_to_cpu)
        self.onload_device = torch.device(onload_device) if onload_device is not None else None
        self._can_move_model = (
            not hasattr(self.model, "hf_device_map")
            and not getattr(self.model, "is_loaded_in_4bit", False)
            and not getattr(self.model, "is_loaded_in_8bit", False)
        )

        if not is_quantized:
            if device is not None:
                self.model = self.model.to(device)
            if dtype is not None:
                self.model = self.model.to(dtype=dtype)
        self.model.eval()

        self.caption_prompt = caption_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self._cached_prompt = None
        self._cached_prompt_text = None

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        self._cached_prompt = prompt
        self._cached_prompt_text = None

    def _get_tokenizer(self):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.processor
        return tokenizer

    def _build_messages(self, instruction: str, caption: str | None = None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        if caption is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                }
            )
        return messages

    def _prepare_inputs(self, image: torch.Tensor, instruction: str, caption: str | None, add_generation_prompt: bool):
        messages = self._build_messages(instruction, caption=caption)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        # process_vision_info expects the image to be part of messages; it will pick it up.
        # We pass the image separately so it can be injected into messages.
        messages[0]["content"][0]["image"] = image
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        device = image.device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        return inputs

    def _generate_caption(self, image: torch.Tensor, instruction: str) -> str:
        inputs = self._prepare_inputs(image=image, instruction=instruction, caption=None, add_generation_prompt=True)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output_ids[:, prompt_len:]
        tokenizer = self._get_tokenizer()
        caption = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return caption

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        if isinstance(prompt, list):
            instruction = self.caption_prompt
        else:
            instruction = self.caption_prompt or prompt

        caption = self._generate_caption(image=image, instruction=instruction)

        prompt_inputs = self._prepare_inputs(
            image=image, instruction=instruction, caption=None, add_generation_prompt=True
        )
        full_inputs = self._prepare_inputs(
            image=image, instruction=instruction, caption=caption, add_generation_prompt=False
        )

        labels = full_inputs["input_ids"].clone()
        labels[:, : prompt_inputs["input_ids"].shape[-1]] = -100

        outputs = self.model(**full_inputs, labels=labels)
        loss = outputs.loss
        return -loss

    def maybe_onload(self):
        if not self.offload_to_cpu:
            return
        if not self._can_move_model:
            return
        target = self.onload_device
        if target is None:
            return
        try:
            self.model.to(target)
        except Exception as exc:
            warnings.warn(f"Qwen25VLCaptionReward failed to onload to {target}: {exc}")

    def maybe_offload(self):
        if not self.offload_to_cpu:
            return
        if not self._can_move_model:
            return
        try:
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            warnings.warn(f"Qwen25VLCaptionReward failed to offload to CPU: {exc}")


class FlorenceSAMReward:
    """
    Florence-2 + SAM reward.

    Uses Florence-2 open-vocabulary detection to propose boxes for the prompt,
    then refines them with SAM and aggregates SAM scores into a scalar reward.
    """

    def __init__(
        self,
        florence_model_id: str = "microsoft/Florence-2-base",
        sam_model_id: str = "facebook/sam-vit-base",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        florence_task: str = "<OPEN_VOCABULARY_DETECTION>",
        max_new_tokens: int = 256,
        sam_multimask_output: bool = True,
        score_reduce: str = "mean",
    ):
        self.florence_model_id = florence_model_id
        self.sam_model_id = sam_model_id
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.token = token
        self.revision = revision
        self.florence_task = florence_task
        self.max_new_tokens = int(max_new_tokens)
        self.sam_multimask_output = bool(sam_multimask_output)
        self.score_reduce = score_reduce

        # Models are intentionally not loaded here. This is a stub.
        self.florence_model = None
        self.florence_processor = None
        self.sam_model = None
        self.sam_processor = None
        self.model = None

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        # Placeholder for any precomputation or caching.
        return None

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        self._maybe_load()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            image = image.float()
        image = torch.clamp(image, 0, 1)

        model_device = self._model_device()
        if model_device is not None and image.device != model_device:
            image = image.to(model_device)

        pil_images, image_sizes = self._to_pil_batch(image)
        prompts = self._build_prompts(prompt, len(pil_images))

        detections = self._run_florence(pil_images, prompts, image_sizes)
        rewards = []
        for pil_image, det in zip(pil_images, detections):
            boxes, scores = det
            if not boxes:
                rewards.append(torch.tensor(0.0, device=image.device, dtype=image.dtype))
                continue
            reward = self._run_sam(pil_image, boxes, scores, image.device, image.dtype)
            rewards.append(reward)

        return torch.stack(rewards).to(image.device)

    def _maybe_load(self):
        if self.florence_model is not None and self.sam_model is not None:
            return
        try:
            from transformers import AutoProcessor, Florence2ForConditionalGeneration, SamModel, SamProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("FlorenceSAMReward requires transformers with Florence2 and SAM.") from exc

        hf_kwargs = _build_hf_kwargs(
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            token=self.token,
            revision=self.revision,
        )
        hf_model_kwargs = dict(hf_kwargs)
        if self.dtype is not None:
            hf_model_kwargs.setdefault("torch_dtype", self.dtype)

        if self.florence_model is None:
            self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_id, **hf_kwargs)
            self.florence_model = Florence2ForConditionalGeneration.from_pretrained(
                self.florence_model_id, **hf_model_kwargs
            )
        if self.sam_model is None:
            self.sam_processor = SamProcessor.from_pretrained(self.sam_model_id, **hf_kwargs)
            self.sam_model = SamModel.from_pretrained(self.sam_model_id, **hf_model_kwargs)

        if self.dtype is not None:
            self.florence_model = self.florence_model.to(dtype=self.dtype)
            self.sam_model = self.sam_model.to(dtype=self.dtype)
        if self.device is not None:
            self.florence_model = self.florence_model.to(self.device)
            self.sam_model = self.sam_model.to(self.device)

        self.florence_model.eval()
        self.sam_model.eval()
        self.model = self.florence_model

    def _model_device(self) -> torch.device | None:
        if self.florence_model is None:
            return self.device
        try:
            return next(self.florence_model.parameters()).device
        except StopIteration:
            return self.device

    def _to_pil_batch(self, image: torch.Tensor) -> tuple[list[PIL.Image.Image], list[tuple[int, int]]]:
        images = []
        sizes = []
        for img in image.detach():
            arr = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
            pil = PIL.Image.fromarray(arr)
            images.append(pil)
            sizes.append(pil.size)  # (width, height)
        return images, sizes

    def _build_prompts(self, prompt: str | list[str], batch_size: int) -> list[str]:
        if isinstance(prompt, list):
            if len(prompt) == batch_size:
                prompts = prompt
            else:
                prompts = [prompt[0]] * batch_size
        else:
            prompts = [prompt] * batch_size
        task = self.florence_task
        return [f"{task} {p}".strip() if p else task for p in prompts]

    def _run_florence(
        self,
        images: list[PIL.Image.Image],
        prompts: list[str],
        image_sizes: list[tuple[int, int]],
    ) -> list[tuple[list[list[float]], list[float]]]:
        inputs = self.florence_processor(images=images, text=prompts, return_tensors="pt")
        model_device = self._model_device()
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(model_device)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)

        outputs = []
        for text, size in zip(decoded, image_sizes):
            parsed = self.florence_processor.post_process_generation(
                text=text, task=self.florence_task, image_size=size
            ).get(self.florence_task, {})
            boxes, scores = self._extract_boxes_and_scores(parsed, size)
            outputs.append((boxes, scores))
        return outputs

    def _extract_boxes_and_scores(self, parsed: dict, size: tuple[int, int]) -> tuple[list[list[float]], list[float]]:
        width, height = size
        boxes: list[list[float]] = []
        scores: list[float] = []

        if isinstance(parsed, dict):
            for box in parsed.get("bboxes", []) or []:
                boxes.append(list(box))
            if "scores" in parsed:
                scores.extend([float(s) for s in parsed.get("scores", [])])
            if "polygons" in parsed:
                for poly in parsed.get("polygons", []) or []:
                    xs = poly[0::2]
                    ys = poly[1::2]
                    if not xs or not ys:
                        continue
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
                    scores.append(1.0)

        if len(scores) < len(boxes):
            scores.extend([1.0] * (len(boxes) - len(scores)))

        # Clamp + filter invalid boxes
        clean_boxes = []
        clean_scores = []
        for box, score in zip(boxes, scores):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = max(0.0, min(float(x1), width - 1))
            x2 = max(0.0, min(float(x2), width - 1))
            y1 = max(0.0, min(float(y1), height - 1))
            y2 = max(0.0, min(float(y2), height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            clean_boxes.append([x1, y1, x2, y2])
            clean_scores.append(float(score))

        return clean_boxes, clean_scores

    def _run_sam(
        self,
        image: PIL.Image.Image,
        boxes: list[list[float]],
        scores: list[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sam_inputs = self.sam_processor(images=image, input_boxes=[boxes], return_tensors="pt")
        for k, v in list(sam_inputs.items()):
            if torch.is_tensor(v):
                sam_inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = self.sam_model(**sam_inputs, multimask_output=self.sam_multimask_output)

        det_scores = torch.tensor(scores, device=device, dtype=dtype)

        if outputs.iou_scores is not None:
            iou = outputs.iou_scores.squeeze(0)
            if iou.dim() == 2:
                iou = iou.max(dim=-1).values
            reward = self._reduce(iou * det_scores)
            return reward.to(dtype=dtype)

        if outputs.pred_masks is None:
            return torch.tensor(0.0, device=device, dtype=dtype)

        masks = outputs.pred_masks.squeeze(0)
        if masks.dim() == 4:
            masks = masks.max(dim=1).values  # [num_boxes, H, W]
        masks = masks.sigmoid()
        area = masks.mean(dim=(-2, -1))
        reward = self._reduce(area * det_scores)
        return reward.to(dtype=dtype)

    def _reduce(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        if self.score_reduce == "sum":
            return scores.sum()
        if self.score_reduce == "max":
            return scores.max()
        return scores.mean()


class GroundingDINOReward:
    """
    Differentiable Grounding-DINO reward.

    Uses torch-only preprocessing for images and raw logits for a smooth reward.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        token: str | None = None,
        revision: str | None = None,
        text_labels: list[str] | list[list[str]] | None = None,
        token_reduce: str = "max",
        query_reduce: str = "max",
    ):
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("GroundingDINOReward requires transformers.") from exc

        hf_kwargs = _build_hf_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        hf_model_kwargs = dict(hf_kwargs)
        if dtype is not None:
            hf_model_kwargs.setdefault("torch_dtype", dtype)

        self.processor = AutoProcessor.from_pretrained(model_id, **hf_kwargs)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **hf_model_kwargs)
        if device is not None:
            self.model = self.model.to(device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        self.model.eval()
        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = dtype or torch.float32

        self.text_labels = text_labels
        self.token_reduce = token_reduce
        self.query_reduce = query_reduce

        cfg = getattr(self.model, "config", None)
        self.image_size = _get_vision_attr(cfg, "image_size", None)
        self.image_mean = _get_vision_attr(cfg, "image_mean", (0.485, 0.456, 0.406))
        self.image_std = _get_vision_attr(cfg, "image_std", (0.229, 0.224, 0.225))

    def prepare(self, prompt: str | list[str], device: torch.device | None = None):
        return None

    def _build_text_labels(self, prompt: str | list[str], batch_size: int) -> list[list[str]]:
        if self.text_labels is None:
            if isinstance(prompt, list):
                if len(prompt) == batch_size:
                    return [[p] for p in prompt]
                return [[prompt[0]] for _ in range(batch_size)]
            return [[prompt] for _ in range(batch_size)]

        if isinstance(self.text_labels, list) and len(self.text_labels) > 0:
            if isinstance(self.text_labels[0], list):
                labels = self.text_labels  # already list[list[str]]
                if len(labels) == batch_size:
                    return labels
                return [labels[0] for _ in range(batch_size)]
            return [self.text_labels for _ in range(batch_size)]

        return [[str(prompt)] for _ in range(batch_size)]

    def _resize_shortest_edge(self, image: torch.Tensor, target_shortest: int) -> torch.Tensor:
        _, _, h, w = image.shape
        if min(h, w) == target_shortest:
            return image
        scale = float(target_shortest) / float(min(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False)

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            image = image.float()
        image = torch.clamp(image, 0, 1)

        image_processor = getattr(self.processor, "image_processor", None)
        size = None
        if image_processor is not None:
            size = getattr(image_processor, "size", None) or getattr(image_processor, "crop_size", None)

        if isinstance(size, dict):
            target_h = size.get("height")
            target_w = size.get("width")
            shortest = size.get("shortest_edge")
            if target_h is not None and target_w is not None:
                if image.shape[-2:] != (target_h, target_w):
                    image = F.interpolate(image, size=(target_h, target_w), mode="bicubic", align_corners=False)
            elif shortest is not None:
                image = self._resize_shortest_edge(image, int(shortest))
        elif isinstance(size, (int, float)):
            target = int(size)
            if image.shape[-2:] != (target, target):
                image = F.interpolate(image, size=(target, target), mode="bicubic", align_corners=False)
        elif self.image_size is not None:
            target = int(self.image_size)
            if image.shape[-2:] != (target, target):
                image = F.interpolate(image, size=(target, target), mode="bicubic", align_corners=False)

        mean = torch.tensor(self.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std
        if image.dtype != self.model_dtype:
            image = image.to(self.model_dtype)
        return image

    def _reduce_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.token_reduce == "mean":
            scores = scores.mean(dim=-1)
        elif self.token_reduce == "sum":
            scores = scores.sum(dim=-1)
        else:
            scores = scores.max(dim=-1).values

        if self.query_reduce == "mean":
            scores = scores.mean(dim=-1)
        elif self.query_reduce == "sum":
            scores = scores.sum(dim=-1)
        else:
            scores = scores.max(dim=-1).values
        return scores

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        pixel_values = self._preprocess_image(image)
        batch_size = pixel_values.shape[0]
        text_labels = self._build_text_labels(prompt, batch_size)

        text_inputs = self.processor(text=text_labels, return_tensors="pt", padding=True, truncation=True)
        for k, v in list(text_inputs.items()):
            if torch.is_tensor(v):
                v = v.to(pixel_values.device)
                if v.is_floating_point() and v.dtype != self.model_dtype:
                    v = v.to(self.model_dtype)
                text_inputs[k] = v

        device_type = pixel_values.device.type
        if device_type == "cuda" and self.model_dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                outputs = self.model(pixel_values=pixel_values, **text_inputs)
        else:
            outputs = self.model(pixel_values=pixel_values, **text_inputs)
        logits = outputs.logits  # [B, num_queries, num_tokens]
        scores = logits.sigmoid()
        scores = self._reduce_scores(scores)
        return scores


class HPSReward:
    def __init__(self, model_name: str = "v2.1", device: str | None = None):
        try:
            import hpsv2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("HPSReward requires the `hpsv2` package.") from exc

        self.hpsv2 = hpsv2
        self.model_name = model_name
        self.device = device

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        images = torch.clamp(image.detach(), 0, 1)
        prompts = prompt if isinstance(prompt, list) else [prompt] * images.shape[0]

        scores = []
        for img, text in zip(images, prompts):
            arr = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
            pil_img = PIL.Image.fromarray(arr)
            try:
                score = self.hpsv2.score(pil_img, text, model_name=self.model_name, device=self.device)
            except TypeError:
                score = self.hpsv2.score(pil_img, text)
            scores.append(float(score))

        return torch.tensor(scores, device=image.device, dtype=image.dtype).mean()
