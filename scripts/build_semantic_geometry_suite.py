"""Build the preregistered v6 model-generated semantic-geometry probe suite.

This is data preparation only. It invokes the frozen pixel-oracle diagnostic
to produce held-out FLUX outputs and never loads or scores a semantic encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_OUTPUT = "experiments/semantic_embedding_geometry_v6"
FORMAL_PROBE_DIR = "/tmp/st_formal_shared_t4_mask"
SOURCE_ROOT = "/data15/hyp/dataset/kontinuous_kontext/raw/source_images"
PROBE_PROVENANCE = "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC"


CASES = (
    {
        "case_id": "ball_color_seed_20260914",
        "source": f"{SOURCE_ROOT}/source_000000.png",
        "instruction": "Make the weighted training ball blue while preserving its shape, texture, lighting, and background.",
        "seed": 20260914,
        "primitive": "color",
        "comparison_focus": "the visible surface color of the foreground weighted training ball",
        "endpoint_question": "What is the visible surface color of the foreground weighted training ball?",
        "source_answer": "The ball is black.",
        "target_answer": "The ball is blue.",
        "semantic_spec_provenance": "validated historical formal ball spec",
        "existing_probe_dir": FORMAL_PROBE_DIR,
    },
    {
        "case_id": "ball_color_seed_20260915",
        "source": f"{SOURCE_ROOT}/source_000000.png",
        "instruction": "Make the weighted training ball blue while preserving its shape, texture, lighting, and background.",
        "seed": 20260915,
        "primitive": "color",
        "comparison_focus": "the visible surface color of the foreground weighted training ball",
        "endpoint_question": "What is the visible surface color of the foreground weighted training ball?",
        "source_answer": "The ball is black.",
        "target_answer": "The ball is blue.",
        "semantic_spec_provenance": "reused validated historical formal ball spec",
    },
    {
        "case_id": "ball_color_seed_20260916",
        "source": f"{SOURCE_ROOT}/source_000000.png",
        "instruction": "Make the weighted training ball blue while preserving its shape, texture, lighting, and background.",
        "seed": 20260916,
        "primitive": "color",
        "comparison_focus": "the visible surface color of the foreground weighted training ball",
        "endpoint_question": "What is the visible surface color of the foreground weighted training ball?",
        "source_answer": "The ball is black.",
        "target_answer": "The ball is blue.",
        "semantic_spec_provenance": "reused validated historical formal ball spec",
    },
    {
        "case_id": "scene_reimagination_seed_20260914",
        "source": f"{SOURCE_ROOT}/source_000004.png",
        "instruction": "Reimagine the entire scene as a serene, minimalist room with a modern, sleek desk setup, maintaining the ergonomic object's design and placement.",
        "seed": 20260914,
        "primitive": "scene reimagination",
        "comparison_focus": "the type of setting surrounding the main ergonomic fitness object",
        "endpoint_question": "What kind of setting surrounds the main ergonomic fitness object?",
        "source_answer": "A beach at sunrise.",
        "target_answer": "A serene minimalist room with a modern desk setup.",
        "semantic_spec_provenance": "human-audited spec from existing dataset instruction",
        "dataset_instruction_metadata": "/data15/hyp/dataset/kontinuous_kontext_runs/small_batch_qwen_v2_20260626_042947/metadata/edit_pairs.jsonl",
    },
    {
        "case_id": "environment_night_seed_20260914",
        "source": f"{SOURCE_ROOT}/source_000005.png",
        "instruction": "Replace the sunny park setting with a stormy night sky, preserving the fitness tool's position and framing.",
        "seed": 20260914,
        "primitive": "environment",
        "comparison_focus": "the weather and time-of-day of the background environment",
        "endpoint_question": "What environment surrounds the foreground fitness tool?",
        "source_answer": "A sunny daytime park.",
        "target_answer": "A stormy night sky.",
        "semantic_spec_provenance": "human-audited spec from existing dataset instruction",
        "dataset_instruction_metadata": "/data15/hyp/dataset/kontinuous_kontext_runs/small_batch_qwen_v2_20260626_042947/metadata/edit_pairs.jsonl",
    },
)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formal-only", action="store_true")
    return parser.parse_args()


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_paths(directory: Path) -> dict[str, Path]:
    return {
        "source": directory / "source.png",
        "full": directory / "native_full.png",
        "0.2": directory / "best_0p2.png",
        "0.5": directory / "best_0p5.png",
        "0.8": directory / "best_0p8.png",
    }


def _stage_suite_inputs(source_paths: dict[str, Path], directory: Path) -> dict[str, Path]:
    """Copy only the frozen images scored by v6 into a portable suite directory."""

    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "source": "source_INPUT.png",
        "full": "native_full_MODEL_GENERATED.png",
        "0.2": "probe_0p2_PIXEL_ORACLE_MODEL_GENERATED.png",
        "0.5": "probe_0p5_PIXEL_ORACLE_MODEL_GENERATED.png",
        "0.8": "probe_0p8_PIXEL_ORACLE_MODEL_GENERATED.png",
    }
    staged = {key: directory / filename for key, filename in filenames.items()}
    for key, target in staged.items():
        shutil.copy2(source_paths[key], target)
    return staged


def _generate_case(case: dict[str, object], directory: Path, args) -> None:
    paths = _probe_paths(directory)
    if args.reuse_existing and (directory / "report.json").exists() and all(path.exists() for path in paths.values()):
        return
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/validate_kontext_terminal_control.py",
        "--model",
        args.model,
        "--source",
        str(case["source"]),
        "--prompt",
        str(case["instruction"]),
        "--output-dir",
        str(directory),
        "--steps",
        "12",
        "--seed",
        str(case["seed"]),
        "--height",
        "256",
        "--width",
        "256",
        "--guidance-scale",
        "2.5",
        "--dtype",
        "bfloat16",
        "--device",
        args.device,
        "--strengths",
        "0.2",
        "0.5",
        "0.8",
        "--control-steps",
        "4",
        "--control-mode",
        "shared-linear",
        "--control-mask-mode",
        "velocity-topk",
        "--control-mask-topk-fraction",
        "0.25",
        "--outer-iters",
        "20",
        "--control-lr",
        "0.1",
        "--lambda-control",
        "0.0001",
        "--use-checkpointing",
    ]
    subprocess.run(command, check=True)


def main():
    args = _args()
    output_dir = Path(args.output_dir)
    probe_root = output_dir / "model_generated_probes"
    records = []
    selected_cases = CASES[:1] if args.formal_only else CASES
    for case in selected_cases:
        existing = case.get("existing_probe_dir")
        directory = Path(existing) if existing else probe_root / str(case["case_id"])
        if existing is None:
            _generate_case(case, directory, args)
        paths = _probe_paths(directory)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Case {case['case_id']} is missing probes: {missing}")
        paths = _stage_suite_inputs(paths, output_dir / "suite_inputs" / str(case["case_id"]))
        record = {key: value for key, value in case.items() if key != "existing_probe_dir"}
        record.update(
            {
                # Keep default-output manifests portable inside a repository clone.
                "source_path": str(paths["source"]),
                "full_path": str(paths["full"]),
                "probe_paths": {strength: str(paths[strength]) for strength in ("0.2", "0.5", "0.8")},
                "source_fingerprint_sha256": _fingerprint(paths["source"]),
                "full_fingerprint_sha256": _fingerprint(paths["full"]),
                "probe_fingerprints_sha256": {
                    strength: _fingerprint(paths[strength]) for strength in ("0.2", "0.5", "0.8")
                },
                "source_provenance": "SOURCE INPUT",
                "full_provenance": "MODEL-GENERATED NATIVE FULL",
                "probe_provenance": PROBE_PROVENANCE,
            }
        )
        records.append(record)
    manifest = {
        "suite_version": "semantic_embedding_geometry_v6_preregistered",
        "case_selection_frozen_before_encoder_scores": True,
        "controller_run": False,
        "controller_integrated_or_optimized_in_v6": False,
        "pixel_oracle_controller_used_for_probe_data_preparation_only": True,
        "pre_geometry_rejections": [
            {
                "case_id": "object_texture_seed_20260914",
                "reason": "Qwen Source/Full endpoint semantic audit failed before CLIP/SigLIP suite geometry.",
                "replacement_case_id": "scene_reimagination_seed_20260914",
                "encoder_geometry_inspected_before_replacement": False,
            }
        ],
        "probe_generator_config": {
            "steps": 12,
            "strengths": [0.2, 0.5, 0.8],
            "control_steps": 4,
            "control_mode": "shared-linear",
            "control_mask_mode": "velocity-topk",
            "control_mask_topk_fraction": 0.25,
            "outer_iters": 20,
            "control_lr": 0.1,
            "lambda_control": 0.0001,
            "guidance_scale": 2.5,
            "dtype": "bfloat16",
        },
        "cases": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"case_count": len(records), "controller_run": False, "manifest": str(output_dir / "suite_manifest.json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
