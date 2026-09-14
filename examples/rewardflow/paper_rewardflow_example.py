"""Minimal opt-in examples for the auditable paper-faithful RewardFlow path."""

import os

import torch
from PIL import Image

from diffusers import FluxRewardFlowPipeline
from diffusers.pipelines.rewardflow import PaperRewardFlowConfig, Qwen25VQAReward


model_dir = os.environ["REWARDFLOW_MODEL_DIR"]
source_path = os.environ["REWARDFLOW_SOURCE_IMAGE"]
device = "cuda"

pipe = FluxRewardFlowPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(device)
source = Image.open(source_path).convert("RGB")
prompt = "Add black sunglasses while preserving identity, lighting, and composition."

# The paper does not disclose these values. Requiring environment variables
# prevents this example from presenting invented numbers as paper defaults.
paper_config = PaperRewardFlowConfig(
    enabled=True,
    lambda_reward=float(os.environ["REWARDFLOW_LAMBDA_REWARD"]),
    use_kl=True,
    lambda_kl=1.5,
    use_sde_noise=True,
    gamma_min=float(os.environ["REWARDFLOW_GAMMA_MIN"]),
    gamma_max=float(os.environ["REWARDFLOW_GAMMA_MAX"]),
    gamma_rho=float(os.environ["REWARDFLOW_GAMMA_RHO"]),
    collect_trace=True,
    static_reward_weights={"siglip": 1.0},
)

image = pipe(
    image=source,
    prompt=prompt,
    num_inference_steps=35,
    generator=torch.Generator(device=device).manual_seed(42),
    paper_config=paper_config,
).images[0]
image.save("paper_rewardflow_siglip.png")
print(f"Collected {len(pipe.last_paper_trace)} paper-sampler trace records.")


# Experimental Qwen2.5-VL VQA example. The torch-native image path has a
# gradient integration test, but the paper does not disclose margin values.
if os.environ.get("REWARDFLOW_RUN_QWEN_VQA") == "1":
    vqa = Qwen25VQAReward(
        question="What is on the person's face?",
        answer="Black sunglasses.",
        model_id=os.environ.get("REWARDFLOW_QWEN_VQA_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        margin=float(os.environ["REWARDFLOW_VQA_MARGIN"]),
        lambda_margin=float(os.environ["REWARDFLOW_VQA_LAMBDA_MARGIN"]),
        device=device,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    vqa_config = PaperRewardFlowConfig(
        enabled=True,
        lambda_reward=float(os.environ["REWARDFLOW_LAMBDA_REWARD"]),
        use_kl=True,
        lambda_kl=1.5,
        static_reward_weights={"vqa": 1.0},
    )
    vqa_image = pipe(
        image=source,
        prompt=prompt,
        num_inference_steps=35,
        generator=torch.Generator(device=device).manual_seed(42),
        reward_fns={"vqa": vqa},
        paper_config=vqa_config,
    ).images[0]
    vqa_image.save("paper_rewardflow_qwen_vqa.png")
