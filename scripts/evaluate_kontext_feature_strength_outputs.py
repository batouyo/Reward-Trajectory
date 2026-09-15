"""Independently evaluate frozen Best feature-controller outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.feature_controller_evaluation import (
    BLIND_LABELS,
    EVALUATION_MASK_PROVENANCE,
    PRESERVATION_CATEGORIES,
    TRAINING_STRENGTHS,
    VISUAL_JUDGE_PERMUTATION_SEED,
    VISUAL_JUDGE_PROMPT_VERSION,
    build_blind_permutations,
    endpoint_difference_evaluation_mask,
    endpoint_pixel_diagnostics,
    format_strength_tag,
    parse_visual_judge_json,
    remap_visual_judgment,
    summarize_visual_judgments,
    visual_judge_json_schema,
)


DEFAULT_OUTPUT = "experiments/feature_reward_controller_v5"
DEFAULT_BASE_URL = "https://tianyuai.lol/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
EDIT_INSTRUCTION = (
    "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."
)
COMPARISON_FOCUS = "the visible surface color of the foreground weighted training ball"


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.getenv("TIANYUAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("TIANYUAI_VISUAL_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Generate fixed-mask diagnostics and blinded grid without calling the visual judge.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    required = [
        "feature_controller_report.json",
        "source_INPUT.png",
        "native_full_MODEL_GENERATED.png",
        *[f"best_s{format_strength_tag(value)}_MODEL_GENERATED.png" for value in TRAINING_STRENGTHS],
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        parser.error(f"Missing controller artifacts: {missing}")
    if not args.offline_only and not os.getenv("TIANYUAI_API_KEY"):
        parser.error("Set TIANYUAI_API_KEY for the independent visual judge.")
    return args


def _serialize(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(_serialize(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_serialize(rows))


def _tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _judge_prompt() -> str:
    return f"""Given the editing instruction and the five labeled images, consider ONLY the requested semantic change.

Editing instruction:
{EDIT_INSTRUCTION}

Requested semantic change:
{COMPARISON_FOCUS}

Rank Image A through Image E from the least amount of the requested semantic edit to the most amount of the requested semantic edit. Use only visible evidence. Do not estimate percentages.

For each image, report whether it is closer to the source endpoint, intermediate, or closer to the full-edit endpoint. Separately report obvious preservation violations using only these categories: {", ".join(PRESERVATION_CATEGORIES)}. Use an empty list when none are obvious. Also list genuinely indistinguishable image pairs. Do not infer edit amounts from image position or labels."""


def _make_blind_grid(paths, mapping, path):
    panels = [(Image.open(paths[identity]).convert("RGB"), label) for label, identity in mapping.items()]
    width, height = panels[0][0].size
    top = 52
    grid = Image.new("RGB", (width * len(panels), height + top), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 4), "BLINDED VISUAL EVALUATION — NO STRENGTH LABELS", fill="red")
    for index, (image, label) in enumerate(panels):
        draw.text((index * width + 4, 28), f"Image {label}", fill="black")
        grid.paste(image, (index * width, top))
    grid.save(path)


def _low_level_diagnostics(output_dir, paths):
    source = _tensor(paths["source"])
    full = _tensor(paths["native_full"])
    mask = endpoint_difference_evaluation_mask(source, full, top_fraction=0.25)
    Image.fromarray(mask[0, 0].mul(255).byte().numpy(), mode="L").save(
        output_dir / "endpoint_difference_eval_mask.png"
    )
    rows = []
    for strength in TRAINING_STRENGTHS:
        image = _tensor(paths[f"best_{strength:.1f}"])
        rows.append(
            {
                "set": "best_training_output",
                "strength": strength,
                **{
                    name: float(value) for name, value in endpoint_pixel_diagnostics(image, source, full, mask).items()
                },
            }
        )
    for index in range(11):
        strength = index / 10
        image = _tensor(output_dir / f"dense_s{format_strength_tag(strength)}_MODEL_GENERATED.png")
        rows.append(
            {
                "set": "dense_held_out_evaluation",
                "strength": strength,
                **{
                    name: float(value) for name, value in endpoint_pixel_diagnostics(image, source, full, mask).items()
                },
            }
        )
    _csv(output_dir / "low_level_endpoint_diagnostics.csv", rows)
    return mask, rows


def _call_judge(client, model, mapping, paths):
    content = []
    for label in BLIND_LABELS:
        content.extend(
            [
                {"type": "text", "text": f"Image {label}:"},
                {"type": "image_url", "image_url": {"url": _image_data_url(paths[mapping[label]])}},
            ]
        )
    content.append({"type": "text", "text": _judge_prompt()})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "feature_strength_blind_visual_judgment",
                "strict": True,
                "schema": visual_judge_json_schema(),
            },
        },
    )
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError("TianyuAI visual judge returned no assistant message.") from error
    return parse_visual_judge_json(text)


def _majority_relation(summary, identity):
    counts = summary["relation_counts"][identity]
    if not counts:
        return None
    return Counter(counts).most_common(1)[0][0]


def main():
    args = _args()
    output_dir = Path(args.output_dir)
    paths = {
        "source": output_dir / "source_INPUT.png",
        "best_0.2": output_dir / "best_s0p2_MODEL_GENERATED.png",
        "best_0.5": output_dir / "best_s0p5_MODEL_GENERATED.png",
        "best_0.8": output_dir / "best_s0p8_MODEL_GENERATED.png",
        "native_full": output_dir / "native_full_MODEL_GENERATED.png",
    }
    permutations = build_blind_permutations(count=6, seed=VISUAL_JUDGE_PERMUTATION_SEED)
    # Persist the preregistered mappings before making or reading any judge call.
    _json(
        output_dir / "visual_judge_permutations.json",
        {
            "seed": VISUAL_JUDGE_PERMUTATION_SEED,
            "prompt_version": VISUAL_JUDGE_PROMPT_VERSION,
            "permutations": permutations,
            "not_sent_to_judge": "true identities and filenames",
        },
    )
    _make_blind_grid(paths, permutations[0], output_dir / "BLINDED_VISUAL_EVALUATION_GRID.png")
    mask, low_level_rows = _low_level_diagnostics(output_dir, paths)
    report_path = output_dir / "feature_controller_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["low_level_endpoint_evaluation"] = {
        "provenance": EVALUATION_MASK_PROVENANCE,
        "mask_top_fraction": 0.25,
        "mask_active_fraction": float(mask.mean()),
        "candidate_dependent": False,
        "used_by_reward_or_controller": False,
        "rows_file": "low_level_endpoint_diagnostics.csv",
    }
    if args.offline_only:
        report["visual_weak_mid_strong"] = "INCONCLUSIVE"
        report["independent_visual_judge"] = {"status": "NOT_RUN_OFFLINE_ONLY"}
        _json(report_path, report)
        print("OFFLINE_EVALUATION_COMPLETE_VISUAL_JUDGE_NOT_RUN", flush=True)
        return

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["TIANYUAI_API_KEY"], base_url=args.base_url.rstrip("/"))
    trials = []
    for index, mapping in enumerate(permutations):
        payload = _call_judge(client, args.model, mapping, paths)
        remapped = remap_visual_judgment(payload, mapping)
        trials.append(
            {
                "trial": index,
                "blind_mapping": mapping,
                "parsed_judgment": payload,
                **remapped,
            }
        )
    summary = summarize_visual_judgments(trials)
    judge_report = {
        "provenance": {
            "provider": "tianyuai",
            "base_url": args.base_url.rstrip("/"),
            "model": args.model,
            "prompt_version": VISUAL_JUDGE_PROMPT_VERSION,
            "permutation_seed": VISUAL_JUDGE_PERMUTATION_SEED,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "credential_saved": False,
            "used_for_reward_optimization_or_checkpoint_selection": False,
        },
        "prompt": _judge_prompt(),
        "trials": trials,
        "summary": summary,
    }
    _json(output_dir / "independent_visual_judge.json", judge_report)
    report["independent_visual_judge"] = judge_report
    report["visual_weak_mid_strong"] = summary["visual_weak_mid_strong"]
    report["reward_hacking"] = summary["reward_hacking"]
    report["perceptual_percentage_calibration"] = "NOT_ESTABLISHED"
    report["primary_training_visual_table"] = {
        str(strength): {
            "feature_coordinate": report["training"]["best"][str(strength)]["feature_coordinate"],
            "gpt_mean_rank_zero_based": summary["mean_rank_position_zero_based"][f"best_{strength:.1f}"],
            "gpt_endpoint_relation_consensus": _majority_relation(summary, f"best_{strength:.1f}"),
            "pixel_projection": next(
                row["endpoint_pixel_axis_projection"]
                for row in low_level_rows
                if row["set"] == "best_training_output" and row["strength"] == strength
            ),
            "preserve_error_to_source": next(
                row["preserve_mse_to_source"]
                for row in low_level_rows
                if row["set"] == "best_training_output" and row["strength"] == strength
            ),
        }
        for strength in TRAINING_STRENGTHS
    }
    _json(report_path, report)
    print(
        json.dumps(
            {
                "controller_feature_drive": report["controller_feature_drive"],
                "visual_weak_mid_strong": report["visual_weak_mid_strong"],
                "reward_hacking": report["reward_hacking"],
                "perceptual_percentage_calibration": "NOT_ESTABLISHED",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
