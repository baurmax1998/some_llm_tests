"""Export helpers for structurally pruned Qwen3 SwiGLU channels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from src.llm.backend import DEFAULT_MODEL


CUSTOM_MODEL_SOURCE = '''"""Qwen3 with a separately sized MLP in every transformer layer."""
import json
from pathlib import Path

from mlx_lm.models.qwen3 import MLP, Model as Qwen3Model, ModelArgs


class Model(Qwen3Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        manifest_path = Path(__file__).with_name("pruning_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sizes = manifest["intermediate_sizes"]
        if len(sizes) != len(self.model.layers):
            raise ValueError("Pruning-Manifest und Layeranzahl passen nicht zusammen")
        for layer, intermediate_size in zip(self.model.layers, sizes):
            layer.mlp = MLP(args.hidden_size, intermediate_size)
'''


RUNNER_SOURCE = '''"""Classify one text with the structurally pruned local model."""
import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

DESCRIPTIONS = """- sport: Wettkampf, Training und sportliche Leistungen
- kultur: Kunst, Literatur, Musik, Theater und kulturelles Erbe
- technik: Computer, Software, Geräte, Maschinen und technische Entwicklungen
- test: Prüfungen, Tests und Qualitätssicherung"""
PROMPT = """Ordne den folgenden Kontext genau einer Kategorie zu.

Antworte ausschließlich mit der zugehörigen einzelnen Ziffer:
- 0 = sport: Wettkampf, Training und sportliche Leistungen
- 1 = kultur: Kunst, Literatur, Musik, Theater und kulturelles Erbe
- 2 = technik: Computer, Software, Geräte, Maschinen und technische Entwicklungen
- 3 = test: Prüfungen, Tests und Qualitätssicherung

Entscheide nach dem Hauptthema des Kontexts, auch wenn einzelne Wörter zu einer
anderen Kategorie passen. Keine Erklärung, kein JSON und kein zusätzlicher Text.

Kontext: {context}
Antwort:
"""
SYSTEM = (
    "Du bist ein Klassifikator. Befolge exakt das verlangte Ausgabeformat und "
    "antworte ohne Erklärung."
)
CATEGORY_BY_CODE = {"0": "sport", "1": "kultur", "2": "technik", "3": "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context", help="Text, der klassifiziert werden soll")
    args = parser.parse_args()
    model_path = Path(__file__).resolve().parent
    model, tokenizer = load(str(model_path), trust_remote_code=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(context=args.context)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    logits = model(mx.array(input_ids)[None])[:, -1, :]
    codes = tuple(CATEGORY_BY_CODE)
    token_ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in codes]
    selected_logits = logits[:, mx.array(token_ids)]
    probabilities = mx.softmax(selected_logits, axis=-1)
    selected = int(mx.argmax(selected_logits, axis=-1).item())
    print(CATEGORY_BY_CODE[codes[selected]])
    print(f"confidence={float(probabilities[0, selected].item()):.2%}")


if __name__ == "__main__":
    main()
'''


def align_channels_for_quantized_export(
    selectivity_scores: np.ndarray,
    pruning_ratio: float,
    *,
    group_size: int,
) -> set[tuple[int, int]]:
    """Select low-score channels with a group-aligned count in every layer."""
    if selectivity_scores.ndim != 2:
        raise ValueError("selectivity_scores muss (Layer, Kanäle) enthalten")
    if not 0 <= pruning_ratio < 1:
        raise ValueError("pruning_ratio muss zwischen 0 und 1 liegen")

    layer_count, channels_per_layer = selectivity_scores.shape
    target_total = round(selectivity_scores.size * pruning_ratio)
    flat_order = np.argsort(selectivity_scores.reshape(-1), kind="stable")
    target_indices = flat_order[:target_total]
    target_layers = np.unravel_index(target_indices, selectivity_scores.shape)[0]
    target_counts = np.bincount(target_layers, minlength=layer_count)

    # down_proj wird entlang seiner Eingabedimension quantisiert. Deshalb muss
    # die verbleibende (und bei gleicher Ausgangsgröße auch die entfernte)
    # Kanalzahl pro Layer durch die Quantisierungsgruppengröße teilbar sein.
    aligned_counts = (target_counts // group_size) * group_size
    aligned_counts = np.minimum(aligned_counts, channels_per_layer - group_size)
    channels: set[tuple[int, int]] = set()
    for layer_index, remove_count in enumerate(aligned_counts):
        if remove_count == 0:
            continue
        order = np.argsort(selectivity_scores[layer_index], kind="stable")
        channels.update(
            (layer_index, int(channel)) for channel in order[:remove_count]
        )

    return channels


def _prune_linear(linear, kept_channels: np.ndarray, *, prune_output: bool):
    import mlx.core as mx
    import mlx.nn as nn

    has_bias = hasattr(linear, "bias")
    if hasattr(linear, "scales"):
        weight = mx.dequantize(
            linear.weight,
            linear.scales,
            linear.biases,
            linear.group_size,
            linear.bits,
            linear.mode,
        )
        selected = mx.array(kept_channels)
        weight = weight[selected] if prune_output else weight[:, selected]
        quantized = mx.quantize(
            weight,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
        mx.eval(*quantized)
        output_dims, input_dims = weight.shape
        # Die Dimensionen werden durch die anschließend gesetzten Tensoren
        # bestimmt. Eine minimale Initialisierung vermeidet große Zufallsmatrizen.
        replacement = nn.QuantizedLinear(
            linear.group_size,
            1,
            bias=has_bias,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
        replacement.weight, replacement.scales, *biases = quantized
        replacement.biases = biases[0] if biases else None
    else:
        selected = mx.array(kept_channels)
        weight = linear.weight[selected] if prune_output else linear.weight[:, selected]
        mx.eval(weight)
        output_dims, input_dims = weight.shape
        replacement = nn.Linear(1, 1, bias=has_bias)
        replacement.weight = weight

    if has_bias:
        replacement.bias = linear.bias[kept_channels] if prune_output else linear.bias
    return replacement


def export_structurally_pruned_qwen3(
    backend,
    removed_channels: Iterable[tuple[int, int]],
    destination: str | Path,
    *,
    pruning_ratio: float,
    accuracy: float,
    distinct_classes: int,
) -> Path:
    """Physically shrink Qwen3 MLP matrices and save an executable local copy."""
    from mlx_lm.utils import hf_repo_to_path, load_config, save_config, save_model

    destination = Path(destination)
    removed_by_layer: dict[int, set[int]] = {}
    for layer_index, channel_index in removed_channels:
        removed_by_layer.setdefault(layer_index, set()).add(channel_index)

    transformer = backend.model.model
    original_size = int(transformer.args.intermediate_size)
    intermediate_sizes = []
    removed_manifest = {}
    for layer_index, layer in enumerate(transformer.layers):
        removed = sorted(removed_by_layer.get(layer_index, ()))
        removed_set = set(removed)
        kept = np.array(
            [index for index in range(original_size) if index not in removed_set],
            dtype=np.int32,
        )
        layer.mlp.gate_proj = _prune_linear(
            layer.mlp.gate_proj, kept, prune_output=True
        )
        layer.mlp.up_proj = _prune_linear(
            layer.mlp.up_proj, kept, prune_output=True
        )
        layer.mlp.down_proj = _prune_linear(
            layer.mlp.down_proj, kept, prune_output=False
        )
        intermediate_sizes.append(int(kept.size))
        removed_manifest[str(layer_index)] = removed

    model_id = os.getenv("TEST_MODEL_ID", DEFAULT_MODEL)
    source_path = Path(model_id)
    if not source_path.exists():
        source_path = hf_repo_to_path(model_id)
    config = load_config(source_path)
    config["model_file"] = "model.py"
    config["structural_pruning"] = {
        "source_model": model_id,
        "selection": "lowest_category_selectivity",
        "requested_ratio": pruning_ratio,
        "actual_removed_channels": sum(len(v) for v in removed_by_layer.values()),
        "accuracy": accuracy,
        "distinct_classes": distinct_classes,
    }

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    save_model(destination, backend.model)
    save_config(config, destination / "config.json")
    backend.tokenizer.save_pretrained(destination)
    generation_config = source_path / "generation_config.json"
    if generation_config.exists():
        shutil.copy2(generation_config, destination / generation_config.name)

    manifest = {
        "intermediate_sizes": intermediate_sizes,
        "removed_channels": removed_manifest,
        **config["structural_pruning"],
    }
    (destination / "pruning_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (destination / "model.py").write_text(CUSTOM_MODEL_SOURCE, encoding="utf-8")
    (destination / "run_pruned.py").write_text(RUNNER_SOURCE, encoding="utf-8")
    (destination / "README.md").write_text(
        "# Structurally pruned Qwen3 categorization model\n\n"
        "Run from the project root:\n\n"
        "```bash\n"
        ".venv/bin/python models/qwen3-categorization-pruned/run_pruned.py "
        "\"Der FC gewinnt das Fußballspiel.\"\n"
        "```\n\n"
        "Loading this custom architecture requires `trust_remote_code=True`.\n",
        encoding="utf-8",
    )
    return destination
