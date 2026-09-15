"""Parse and cache a Source/Native-Full semantic specification with OpenAI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from diffusers.pipelines.rewardflow.relative_endpoint_parser import (
    OpenAIRelativeEndpointParser,
    load_cached_relative_endpoint_parse,
    save_cached_relative_endpoint_parse,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()

    online = OpenAIRelativeEndpointParser(model=args.model)
    from diffusers.pipelines.rewardflow.relative_endpoint_parser import fingerprint_endpoint_image

    source_fingerprint = fingerprint_endpoint_image(args.source)
    full_fingerprint = fingerprint_endpoint_image(args.full)
    record = load_cached_relative_endpoint_parse(
        args.cache,
        source_fingerprint,
        full_fingerprint,
        args.instruction,
        model=online.model,
    )
    cache_hit = record is not None
    if record is None:
        record = online.parse(args.source, args.full, args.instruction)
        save_cached_relative_endpoint_parse(args.cache, record)
    output = {"spec": asdict(record.spec), "provenance": record.provenance, "cache_hit": cache_hit}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SEMANTIC_SPEC_READY cache_hit={str(cache_hit).lower()} output={args.output}")


if __name__ == "__main__":
    main()
