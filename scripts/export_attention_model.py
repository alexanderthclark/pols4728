#!/usr/bin/env python3
"""Export real, reproducible sentiment predictions and every attention head.

Install the pinned requirements in requirements-attention-model.txt, then run:
    python scripts/export_attention_model.py --output docs/attention/data/infinite-jest-attention.json

This performs inference only. It does not train, edit, or select a checkpoint
based on the two examples. The output contains the tokenizer's actual special
tokens and subwords, and preserves the original full attention tensors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
EXAMPLES = (
    ("Good and Infinite Jest don't belong in the same sentence", "Negative"),
    ("Infinite Jest wasn't just good. It was spectacular.", "Positive"),
)


def export(output_path: Path) -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to("cpu").eval()
    config = model.config
    labels = [config.id2label[index] for index in range(config.num_labels)]
    assert labels == ["NEGATIVE", "POSITIVE"], labels
    assert config.n_layers == 6 and config.n_heads == 12
    examples = []

    for text, intended_reading in EXAMPLES:
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        token_ids = inputs["input_ids"][0].tolist()
        with torch.inference_mode():
            result = model(**inputs, output_attentions=True, return_dict=True)
        assert result.attentions is not None
        attention = torch.stack(result.attentions)[:, 0].detach().cpu()
        expected_shape = (config.n_layers, config.n_heads, len(token_ids), len(token_ids))
        assert tuple(attention.shape) == expected_shape, (attention.shape, expected_shape)
        assert torch.isfinite(attention).all() and (attention >= 0).all()
        row_error = (attention.sum(dim=-1) - 1).abs().max().item()
        assert row_error < 2e-6, row_error
        logits = result.logits[0].detach().cpu()
        probabilities = logits.softmax(dim=-1)
        assert torch.isfinite(logits).all() and torch.isfinite(probabilities).all()
        prediction_index = int(probabilities.argmax())
        examples.append({
            "text": text,
            "intended_reading": intended_reading,
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_ids": token_ids,
            "token_offsets": offsets,
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "prediction": {
                "label": labels[prediction_index],
                "index": prediction_index,
                "probability": float(probabilities[prediction_index]),
            },
            "attentions": attention.tolist(),
            "validation": {"maximum_attention_row_sum_error": row_error},
        })

    payload = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "url": f"https://huggingface.co/{MODEL_ID}",
            "revision_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
            "architecture": config.architectures[0],
            "num_layers": config.n_layers,
            "num_heads": config.n_heads,
            "hidden_size": config.dim,
            "labels": labels,
            "readout": "[CLS]",
            "readout_description": "Final-layer [CLS] state, linear pre-classifier, ReLU, dropout (disabled in evaluation), linear classifier, then softmax.",
        },
        "inference": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy"
            )},
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "model_mode": "eval",
            "padding": False,
            "truncation": False,
            "attention_axes": ["layer", "head", "destination_token", "source_token"],
            "indexing": "Arrays are zero-based; UI layer and head numbers may be displayed one-based.",
            "precision": "Unrounded float32 inference results serialized as JSON numbers.",
        },
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    output_path.write_bytes(encoded)
    print(json.dumps({
        "output": str(output_path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "model": MODEL_ID,
        "revision": REVISION,
        "examples": [{key: example[key] for key in (
            "text", "tokens", "logits", "probabilities", "prediction", "validation"
        )} for example in examples],
    }, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
