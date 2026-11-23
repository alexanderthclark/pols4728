#!/usr/bin/env python3
"""Reproduce the measured Infinite Jest investigation used by the lesson.

Install scripts/requirements-attention-model.txt, then run:
  python scripts/export_attention_investigation.py \
    --output docs/attention/data/infinite-jest-investigation.json

An optional --model-path points to an already downloaded pinned checkpoint.
Only the declared examples and selected interventions are run. There is no
training, head search, private input, or dependency on an earlier scratch run.
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
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
WEIGHTS_SHA256 = "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa"
TEXTS = {
    "short": "Infinite Jest wasn't just good.",
    "without_just": "Infinite Jest wasn't good.",
    "full": "Infinite Jest wasn't just good. It was spectacular.",
    "negative": "Good and Infinite Jest don't belong in the same sentence",
    "very": "Infinite Jest wasn't very good.",
}
SUBJECTS = ["Infinite Jest", "Dune", "Middlemarch", "This book", "This novel"]
# These eight pairs were fixed before the separate state-transfer validation.
# The discovery pair (Infinite Jest / good / bad) is excluded from its 16 runs.
VALIDATION_PAIRS = [
    ("dune_good_bad", "Dune", "good", "bad"),
    ("middlemarch_good_bad", "Middlemarch", "good", "bad"),
    ("book_good_bad", "This book", "good", "bad"),
    ("movie_good_bad", "The movie", "good", "bad"),
    ("ij_excellent_terrible", "Infinite Jest", "excellent", "terrible"),
    ("ij_enjoyable_boring", "Infinite Jest", "enjoyable", "boring"),
    ("book_excellent_terrible", "This book", "excellent", "terrible"),
    ("movie_enjoyable_boring", "The movie", "enjoyable", "boring"),
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def score(logits: torch.Tensor) -> dict:
    values = logits.detach().reshape(-1)
    assert values.numel() == 2 and torch.isfinite(values).all()
    return {
        "logits": values.tolist(),
        "margin": float(values[1] - values[0]),
        "positive_probability": float(values.softmax(-1)[1]),
        "prediction": "POSITIVE" if values[1] > values[0] else "NEGATIVE",
    }


class Measurements:
    def __init__(self, model_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, attn_implementation="eager", torch_dtype=torch.float32,
        ).to("cpu").eval()
        assert [self.model.config.id2label[i] for i in range(2)] == ["NEGATIVE", "POSITIVE"]
        assert self.model.config.n_layers == 6 and self.model.config.n_heads == 12
        self.blocks = self.model.distilbert.transformer.layer
        self.cache = {}

    def baseline(self, text: str) -> dict:
        if text not in self.cache:
            inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
            with torch.inference_mode():
                result = self.model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
            self.cache[text] = {"text": text, "inputs": inputs, "tokens": tokens, "result": result, "score": score(result.logits)}
        return self.cache[text]

    def public_baseline(self, text: str) -> dict:
        b = self.baseline(text)
        return {"text": text, "tokens": b["tokens"], "score": b["score"]}

    def run(self, text: str, **kwargs):
        with torch.inference_mode():
            return self.model(**self.baseline(text)["inputs"], output_attentions=True, return_dict=True, **kwargs)

    def two_edges(self, text: str, renormalize: bool = False) -> dict:
        b = self.baseline(text)
        n = len(b["tokens"])
        mask = torch.ones((6, 1, 12, n, n))
        t, just = b["tokens"].index("t"), b["tokens"].index("just")
        a = b["result"].attentions[1][0, 8]
        edges = []
        for destination, source in ((t, just), (just, t)):
            weight = float(a[destination, source])
            if renormalize:
                mask[1, 0, 8, destination, :] = 1 / (1 - weight)
            mask[1, 0, 8, destination, source] = 0
            edges.append({"destination_index": destination, "source_index": source,
                          "destination_token": b["tokens"][destination], "source_token": b["tokens"][source], "weight": weight})
        result = self.run(text, head_mask=mask)
        for edge in edges:
            row = result.attentions[1][0, 8, edge["destination_index"]]
            assert float(row[edge["source_index"]]) == 0
            expected = 1 if renormalize else 1 - edge["weight"]
            assert abs(float(row.sum()) - expected) < 3e-6
        after = score(result.logits)
        return {"text": text, "tokens": b["tokens"], "before": b["score"], "after": after,
                "delta_margin": after["margin"] - b["score"]["margin"], "edges": edges,
                "renormalized": renormalize}

    def remove_message(self, text: str, source_token: str, head: int = 10) -> dict:
        b = self.baseline(text)
        layer, destination = 6, 0
        source = b["tokens"].index(source_token)
        attention = self.blocks[layer - 1].attention
        width = self.model.config.dim // self.model.config.n_heads
        section = slice((head - 1) * width, head * width)
        with torch.inference_mode():
            value = attention.v_lin(b["result"].hidden_states[layer - 1])[0, source, section]
            projected = value @ attention.out_lin.weight[:, section].T
            weight = float(b["result"].attentions[layer - 1][0, head - 1, destination, source])
            message = weight * projected

        def subtract(_module, _args, output):
            changed = output[0].clone()
            changed[0, destination] -= message
            return (changed,) + output[1:]

        handle = attention.register_forward_hook(subtract)
        try:
            after = score(self.run(text).logits)
        finally:
            handle.remove()
        return {"id": "apostrophe" if source_token == "'" else source_token,
                "layer": layer, "head": head, "destination_index": destination,
                "destination_token": "[CLS]", "source_index": source, "source_token": source_token,
                "weight": weight, "value_norm": float(value.norm()),
                "projected_value_norm": float(projected.norm()), "message_norm": float(message.norm()),
                "after": after, "delta_margin": after["margin"] - b["score"]["margin"]}

    def patch_state(self, recipient_text: str, donor_text: str, block: int = 4) -> dict:
        recipient, donor = self.baseline(recipient_text), self.baseline(donor_text)
        i, j = recipient["tokens"].index("not"), donor["tokens"].index("not")
        replacement = donor["result"].hidden_states[block][:, j]

        def patch(_module, _args, output):
            changed = output[-1].clone()
            changed[:, i] = replacement
            return output[:-1] + (changed,)

        handle = self.blocks[block - 1].register_forward_hook(patch)
        try:
            return score(self.run(recipient_text).logits)
        finally:
            handle.remove()

    def patch_values(self, recipient_text: str, donor_text: str) -> tuple[dict, float]:
        recipient, donor = self.baseline(recipient_text), self.baseline(donor_text)
        i, j = recipient["tokens"].index("not"), donor["tokens"].index("not")
        projection = self.blocks[4].attention.v_lin
        with torch.inference_mode():
            replacement = projection(donor["result"].hidden_states[4])[:, j].clone()

        def patch(_module, _args, output):
            changed = output.clone()
            changed[:, i] = replacement
            return changed

        handle = projection.register_forward_hook(patch)
        try:
            result = self.run(recipient_text)
        finally:
            handle.remove()
        error = float((result.attentions[4] - recipient["result"].attentions[4]).abs().max())
        assert error == 0
        return score(result.logits), error


def export(model_path: Path, output_path: Path, reference_path: Path) -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    actual_hash = digest(model_path / "model.safetensors")
    assert actual_hash == WEIGHTS_SHA256, "The local safetensors file is not the pinned checkpoint."
    measurements = Measurements(model_path)
    reference = json.loads(reference_path.read_text())
    assert reference["model"]["revision"] == REVISION
    published_checks = []
    for example in reference["examples"]:
        b = measurements.baseline(example["text"])
        actual = torch.stack(b["result"].attentions)[:, 0]
        logits_error = float((b["result"].logits[0] - torch.tensor(example["logits"])).abs().max())
        attention_error = float((actual - torch.tensor(example["attentions"])).abs().max())
        assert logits_error < 1e-5 and attention_error < 1e-5
        assert b["tokens"] == example["tokens"]
        published_checks.append({"text": example["text"], "max_logit_error": logits_error, "max_attention_error": attention_error})

    short, full = measurements.two_edges(TEXTS["short"]), measurements.two_edges(TEXTS["full"])
    short_baseline = measurements.baseline(TEXTS["short"])
    n = len(short_baseline["tokens"])
    no_op_logits = measurements.run(TEXTS["short"], head_mask=torch.ones((6, 1, 12, n, n))).logits
    no_op_error = float((no_op_logits - short_baseline["result"].logits).abs().max())
    assert no_op_error == 0
    phrase_validation = []
    for subject in SUBJECTS:
        s = measurements.two_edges(f"{subject} wasn't just good.")
        f = measurements.two_edges(f"{subject} wasn't just good. It was spectacular.")
        r = measurements.two_edges(f"{subject} wasn't just good.", renormalize=True)
        phrase_validation.append({"subject": subject, "short": s, "full": f, "renormalized_short": r})

    scope_texts = {"negative": "Infinite Jest was bad, not good.", "positive": "Infinite Jest was good, not bad."}
    context = {"layer": 5, "state_after_block": 4, "source_token": "not", "directions": [], "controls": {}}
    for name, text in scope_texts.items():
        context[name] = measurements.public_baseline(text)
        context[name]["not_index"] = context[name]["tokens"].index("not")
    initial_distance = float((measurements.baseline(scope_texts["negative"])["result"].hidden_states[0][0, context["negative"]["not_index"]] - measurements.baseline(scope_texts["positive"])["result"].hidden_states[0][0, context["positive"]["not_index"]]).norm())
    own_errors, final_errors = [], []
    for recipient, donor in (("negative", "positive"), ("positive", "negative")):
        text, source_text = scope_texts[recipient], scope_texts[donor]
        before = context[recipient]["score"]
        values_after, attention_error = measurements.patch_values(text, source_text)
        state_after = measurements.patch_state(text, source_text)
        own = measurements.patch_state(text, text)
        final = measurements.patch_state(text, source_text, block=6)
        own_errors.append(max(abs(a - b) for a, b in zip(own["logits"], before["logits"])))
        final_errors.append(max(abs(a - b) for a, b in zip(final["logits"], before["logits"])))
        context["directions"].append({"recipient": recipient, "donor": donor, "before": before,
            "values_after": values_after, "state_after": state_after,
            "values_delta_margin": values_after["margin"] - before["margin"],
            "state_delta_margin": state_after["margin"] - before["margin"],
            "attention_max_abs_change": attention_error})
    context["controls"] = {"same_initial_not_state_distance": initial_distance,
                            "own_state_max_logit_error": max(own_errors), "final_not_state_max_logit_error": max(final_errors)}
    assert initial_distance == 0 and max(own_errors + final_errors) < 1e-6

    context_validation = []
    for case_id, subject, positive, negative in VALIDATION_PAIRS:
        pair = {"negative": f"{subject} was {negative}, not {positive}.", "positive": f"{subject} was {positive}, not {negative}."}
        directions = []
        for recipient, donor in (("negative", "positive"), ("positive", "negative")):
            before = measurements.baseline(pair[recipient])["score"]
            after = measurements.patch_state(pair[recipient], pair[donor])
            directions.append({"recipient": recipient, "donor": donor, "before": before, "after": after,
                               "flipped": before["prediction"] != after["prediction"]})
        context_validation.append({"id": case_id, "subject": subject, "negative_text": pair["negative"], "positive_text": pair["positive"], "directions": directions})
    payload = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": REVISION, "architecture": "DistilBertForSequenceClassification",
                  "layers": 6, "heads": 12, "hidden_size": 768, "labels": ["NEGATIVE", "POSITIVE"]},
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(), "generator": "scripts/export_attention_investigation.py",
            "source_paths": ["scripts/export_attention_investigation.py", "docs/attention/data/infinite-jest-attention.json"],
            "upstream_implementation": "https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/distilbert/modeling_distilbert.py",
            "checkpoint_weights_sha256": actual_hash, "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy")},
            "settings": {"device": "cpu", "dtype": "float32", "model_mode": "eval", "attention_implementation": "eager", "threads": 1, "padding": False, "truncation": False},
            "indexing": "Layers and heads are one-based; token indices are zero-based, with [CLS], [SEP], punctuation, and subwords retained.",
            "metric": "margin = positive logit minus negative logit; delta = intervention minus original margin",
            "methods": {
                "two_edges": "L2 H9: set A[t,just] and A[just,t] to zero without renormalization. z'_i=z_i-A_ij v_j. Keep all other messages, shared output bias, residual paths and FFNs; rerun later computation. The two reads use incoming states simultaneously.",
                "renormalized_robustness": "A separate intervention: set the selected edge to zero and divide the remaining entries in that row by 1-A_ij. This reallocates attention mass and is not the same intervention as message subtraction.",
                "brightness": "At L6 H10 [CLS], message m_j=A_[CLS,j](v_j W_O[h]). Values include their affine projection bias; remove exactly m_j from the attention output and rerun the classifier, retaining every other message and the shared output bias.",
                "values_patch": "Replace the layer-5 v_lin output at not across all twelve heads with the donor's value. Recipient Q,K,residual states and all layer-5 attention matrices stay fixed; later layers rerun.",
                "state_patch": "Replace only the contextual state at not after the complete fourth block with its donor counterpart; recompute all later operations. This distinct intervention also changes later Q,K,V and the residual path.",
                "scope": "Examples and routes were selected after inspecting this checkpoint. Eight additional state-transfer pairs were declared before validation; title and renormalization checks remain limited to the listed inputs. These are computation interventions, not additive token attributions or universal semantic roles.",
            },
            "checks": {"published_originals": published_checks, "all_ones_mask_max_logit_error": no_op_error,
                       "removed_edges_zero": True, "intervened_row_sums_match_method": True},
        },
        "behavior": {name: measurements.public_baseline(text) for name, text in TEXTS.items()},
        "phrase": {"layer": 2, "head": 9, "text": TEXTS["short"], "tokens": short_baseline["tokens"],
                   "attention": short_baseline["result"].attentions[1][0, 8].tolist(), "edges": short["edges"],
                   "short": {key: short[key] for key in ("before", "after", "delta_margin")}, "full": full},
        "brightness": {"text": TEXTS["negative"], "tokens": measurements.baseline(TEXTS["negative"])["tokens"],
                       "baseline": measurements.baseline(TEXTS["negative"])["score"],
                       "messages": [measurements.remove_message(TEXTS["negative"], "'"), measurements.remove_message(TEXTS["negative"], "t")]},
        "context": context,
        "validation": {
            "phrase": {"cases": phrase_validation,
                       "short_flips": sum(case["short"]["before"]["prediction"] != case["short"]["after"]["prediction"] for case in phrase_validation),
                       "full_flips": sum(case["full"]["before"]["prediction"] != case["full"]["after"]["prediction"] for case in phrase_validation),
                       "renormalized_short_flips": sum(case["renormalized_short"]["before"]["prediction"] != case["renormalized_short"]["after"]["prediction"] for case in phrase_validation)},
            "context": {"pairs": len(context_validation), "interventions": 2 * len(context_validation),
                        "flips": sum(d["flipped"] for case in context_validation for d in case["directions"]),
                        "cases": context_validation},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output_path.write_bytes(encoded)
    print(json.dumps({"output": str(output_path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
                      "phrase_short_flips": payload["validation"]["phrase"]["short_flips"],
                      "phrase_full_flips": payload["validation"]["phrase"]["full_flips"],
                      "context_flips": payload["validation"]["context"]["flips"],
                      "context_interventions": payload["validation"]["context"]["interventions"]}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Already downloaded pinned checkpoint; otherwise download the pinned revision.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "docs/attention/data/infinite-jest-attention.json")
    args = parser.parse_args()
    model_path = args.model_path or Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt", "tokenizer.json", "special_tokens_map.json"]))
    export(model_path, args.output, args.reference)
