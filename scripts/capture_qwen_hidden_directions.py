#!/usr/bin/env python3
"""Capture Qwen hidden-state mean-difference direction diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def parse_layers(value: str, num_layers: int) -> list[int]:
    if value == "all":
        return list(range(num_layers + 1))
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    bad = [layer for layer in layers if layer < 0 or layer > num_layers]
    if bad:
        raise SystemExit(f"Layer index out of range 0..{num_layers}: {bad}")
    return sorted(layers)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_items(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_group.setdefault(item["group"], []).append(item)
    groups = sorted(by_group)
    if max_items < len(groups):
        raise SystemExit(f"--max-items must be >= number of groups ({len(groups)})")
    selected: list[dict[str, Any]] = []
    base = max_items // len(groups)
    remainder = max_items % len(groups)
    for index, group in enumerate(groups):
        take = base + (1 if index < remainder else 0)
        selected.extend(by_group[group][:take])
    return selected


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def cosine(a: Any, b: Any) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0:
        return 0.0
    return float((a @ b).item() / denom)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device-map", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    started = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    corpus = load_json(args.corpus)
    items = select_items(corpus.get("items", []), args.max_items)
    if not items:
        raise SystemExit("No corpus items selected")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "torch_dtype": dtype_map[args.torch_dtype],
    }
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not model_kwargs.get("device_map"):
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model.to(device)
    model.eval()
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    selected_layers = parse_layers(args.layers, num_layers)

    sums: dict[str, dict[int, Any]] = {
        "contested_positive": {},
        "over_contested_settled_negative": {},
    }
    features: dict[str, dict[int, list[Any]]] = {
        "contested_positive": {},
        "over_contested_settled_negative": {},
    }
    counts = {key: 0 for key in sums}
    item_rows = []

    for item in items:
        group = item["group"]
        if group not in sums:
            raise SystemExit(f"Unexpected group {group!r} in {item.get('id')}")
        prompt = render_prompt(tokenizer, item["messages"])
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
            last_index = int(attention_mask.sum().item() - 1)
        else:
            last_index = int(input_ids.shape[1] - 1)

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        counts[group] += 1
        for layer in selected_layers:
            vector = output.hidden_states[layer][0, last_index, :].detach().float().cpu()
            sums[group][layer] = sums[group].get(layer, torch.zeros_like(vector)) + vector
            features[group].setdefault(layer, []).append(vector)
        item_rows.append(
            {
                "id": item["id"],
                "group": group,
                "tokens": int(input_ids.shape[1]),
                "truncated": int(input_ids.shape[1]) >= args.max_length,
            }
        )

    missing_groups = [group for group, count in counts.items() if count == 0]
    if missing_groups:
        raise SystemExit(f"Missing required groups: {missing_groups}")

    layer_rows = []
    direction_tensors: dict[int, Any] = {}
    for layer in selected_layers:
        positive_mean = sums["contested_positive"][layer] / counts["contested_positive"]
        negative_mean = (
            sums["over_contested_settled_negative"][layer]
            / counts["over_contested_settled_negative"]
        )
        direction = negative_mean - positive_mean
        direction_tensors[layer] = direction
        positive_norm = float(positive_mean.norm().item())
        negative_norm = float(negative_mean.norm().item())
        direction_norm = float(direction.norm().item())
        positive_scores = [float((vector @ direction).item()) for vector in features["contested_positive"][layer]]
        negative_scores = [
            float((vector @ direction).item())
            for vector in features["over_contested_settled_negative"][layer]
        ]
        pos_mean_score = sum(positive_scores) / len(positive_scores)
        neg_mean_score = sum(negative_scores) / len(negative_scores)
        pos_var = sum((value - pos_mean_score) ** 2 for value in positive_scores) / max(len(positive_scores) - 1, 1)
        neg_var = sum((value - neg_mean_score) ** 2 for value in negative_scores) / max(len(negative_scores) - 1, 1)
        pooled_sd = math.sqrt((pos_var + neg_var) / 2) if (pos_var + neg_var) > 0 else 0.0
        separation_z = (neg_mean_score - pos_mean_score) / pooled_sd if pooled_sd else 0.0
        layer_rows.append(
            {
                "layer": layer,
                "positive_mean_norm": round(positive_norm, 6),
                "negative_mean_norm": round(negative_norm, 6),
                "direction_norm": round(direction_norm, 6),
                "mean_cosine": round(cosine(positive_mean, negative_mean), 6),
                "positive_projection_mean": round(pos_mean_score, 6),
                "negative_projection_mean": round(neg_mean_score, 6),
                "projection_separation_z": round(separation_z, 6),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction_path = args.output_dir / "directions.pt"
    torch.save(
        {
            "model": args.model,
            "layers": selected_layers,
            "hidden_size": hidden_size,
            "counts": counts,
            "positive_group": "contested_positive",
            "negative_group": "over_contested_settled_negative",
            "direction_sign": "negative_mean - positive_mean",
            "directions": direction_tensors,
        },
        direction_path,
    )
    summary = {
        "schema_version": 1,
        "model": args.model,
        "created_at_unix": int(time.time()),
        "elapsed_s": round(time.time() - started, 3),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layers": selected_layers,
        "counts": counts,
        "max_length": args.max_length,
        "max_items": args.max_items,
        "device_map": args.device_map,
        "device": args.device,
        "direction_path": str(direction_path),
        "top_layers_by_direction_norm": sorted(
            layer_rows,
            key=lambda row: row["direction_norm"],
            reverse=True,
        )[:10],
        "top_layers_by_projection_separation": sorted(
            layer_rows,
            key=lambda row: abs(row["projection_separation_z"]),
            reverse=True,
        )[:10],
        "layers_summary": layer_rows,
        "items": item_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
