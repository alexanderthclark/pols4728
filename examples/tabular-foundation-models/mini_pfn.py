"""Train a small educational PFN on two bear-survival worlds.

This is an independently trained teaching model, not TabPFN. A task has one
hidden world, four labeled context rows, and one unlabeled query row. Neither
world identity nor a hand-engineered bear/strategy interaction enters the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MATCH_PROBABILITY = 1 / (1 + math.exp(-1))


class MiniPFN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.width = width
        self.context_encoder = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width))
        self.query_encoder = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width))
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, context: torch.Tensor, query: torch.Tensor, trace: bool = False):
        encoded_context = self.context_encoder(context)
        encoded_query = self.query_encoder(query)
        q = self.to_q(encoded_query)
        k = self.to_k(encoded_context)
        v = self.to_v(encoded_context)
        scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.width)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, v)
        logit = self.decoder(torch.cat([encoded_query, pooled], dim=-1)).squeeze(-1)
        if not trace:
            return logit
        return {
            "encoded_context": encoded_context, "encoded_query": encoded_query,
            "q": q, "k": k, "v": v, "scores": scores,
            "attention": weights, "weighted_values": weights.unsqueeze(-1) * v,
            "pooled_value": pooled, "logit": logit, "probability": torch.sigmoid(logit),
        }


def sample_tasks(count: int, generator: torch.Generator, n_context: int = 4):
    # Draw a world ONCE per task. It governs all context outcomes and the query.
    world_b = torch.randint(0, 2, (count, 1), generator=generator)
    features = torch.randint(0, 2, (count, n_context + 1, 2), generator=generator).float()
    match = features[..., 0] == features[..., 1]
    logit = (2 * match.float() - 1) * (1 - 2 * world_b)
    probabilities = torch.sigmoid(logit)
    outcomes = (torch.rand((count, n_context + 1), generator=generator) < probabilities).float()
    context = torch.cat([features[:, :n_context], outcomes[:, :n_context, None]], dim=-1)
    return context, features[:, -1], outcomes[:, -1]


def exact_bayes(context: torch.Tensor, query: torch.Tensor):
    """Analytic comparison only; these calculations never enter training inputs."""
    context_match = context[..., 0] == context[..., 1]
    log_odds_a = ((2 * context_match.float() - 1) * (2 * context[..., 2] - 1)).sum(-1)
    posterior_a = torch.sigmoid(log_odds_a)
    query_match = query[:, 0] == query[:, 1]
    probability_in_a = torch.where(query_match, MATCH_PROBABILITY, 1 - MATCH_PROBABILITY)
    prediction = posterior_a * probability_in_a + (1 - posterior_a) * (1 - probability_in_a)
    return prediction, posterior_a


@torch.no_grad()
def evaluate(model, tasks):
    context, query, outcome = tasks
    logit = model(context, query)
    prediction = torch.sigmoid(logit)
    bayes_prediction, _ = exact_bayes(context, query)
    # Scoring against the Bayes conditional distribution removes test-label noise
    # from the excess loss. Observed-label log loss is reported separately.
    expected_loss = F.binary_cross_entropy_with_logits(logit, bayes_prediction).item()
    bayes_entropy = F.binary_cross_entropy(bayes_prediction, bayes_prediction).item()
    return {
        "observed_log_loss": F.binary_cross_entropy_with_logits(logit, outcome).item(),
        "exact_bayes_observed_log_loss": F.binary_cross_entropy(bayes_prediction, outcome).item(),
        "ignore_context_log_loss": math.log(2),
        "expected_log_loss": expected_loss,
        "exact_bayes_expected_log_loss": bayes_entropy,
        "expected_excess_log_loss": expected_loss - bayes_entropy,
        "mean_absolute_probability_error_vs_bayes": torch.mean(torch.abs(prediction - bayes_prediction)).item(),
        "rmse_probability_vs_bayes": torch.sqrt(torch.mean((prediction - bayes_prediction) ** 2)).item(),
    }


@torch.no_grad()
def export_trace(model, labels, name):
    context = torch.tensor([[[0, 0, labels[0]], [0, 1, labels[1]], [1, 0, labels[2]], [1, 1, labels[3]]]], dtype=torch.float32)
    query = torch.tensor([[1, 1]], dtype=torch.float32)
    trace = model(context, query, trace=True)
    bayes_prediction, posterior_a = exact_bayes(context, query)
    result = {
        "name": name,
        "context": context[0].tolist(), "query": query[0].tolist(),
        "context_column_names": ["brown_bear", "lie_down", "survived"],
        "query_column_names": ["brown_bear", "lie_down"],
        "exact_bayes_probability": bayes_prediction.item(),
        "exact_bayes_posterior_a": posterior_a.item(),
    }
    result.update({key: value[0].tolist() for key, value in trace.items()})
    result["row_embeddings"] = result.pop("encoded_context")
    result["query_embedding"] = result.pop("encoded_query")
    result["keys"] = result.pop("k")
    result["values"] = result.pop("v")
    result["pooled"] = result.pop("pooled_value")
    result["illustrative_outcome"] = int(labels[-1])
    result["illustrative_log_loss"] = -math.log(result["probability"] if labels[-1] else 1 - result["probability"])
    assert abs(sum(result["attention"]) - 1) < 1e-6
    reconstructed = torch.tensor(result["weighted_values"]).sum(0)
    assert torch.allclose(reconstructed, trace["pooled_value"][0], atol=1e-6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--test-tasks", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=HERE / "results/mini-pfn-results.json")
    parser.add_argument("--web-output", type=Path, default=REPO / "docs/tabular-foundation-models/mini-pfn-results.mjs")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.test_tasks) < 1 or args.learning_rate <= 0 or args.seed < 0:
        parser.error("steps, batch size, test tasks, and learning rate must be positive; seed must be nonnegative")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    validation_generator = torch.Generator().manual_seed(args.seed + 2)
    test_generator = torch.Generator().manual_seed(args.seed + 3)
    model = MiniPFN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    validation_tasks = sample_tasks(2048, validation_generator)
    checkpoints = [{"step": 0, "validation": evaluate(model, validation_tasks)}]
    recent_loss = 0.0
    first_update = None
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        context, query, outcome = sample_tasks(args.batch_size, train_generator)
        logit = model(context, query)
        loss = F.binary_cross_entropy_with_logits(logit, outcome)
        if step == 1:
            first_update = {
                "context": context[0].tolist(), "query": query[0].tolist(),
                "query_outcome": outcome[0].item(),
                "batch_log_loss_before": loss.item(),
                "first_query_probability_before": torch.sigmoid(logit[0]).item(),
                "optimizer": "Adam", "batch_size": args.batch_size,
            }
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1:
            with torch.no_grad():
                after = model(context, query)
                first_update["batch_log_loss_after"] = F.binary_cross_entropy_with_logits(after, outcome).item()
                first_update["first_query_probability_after"] = torch.sigmoid(after[0]).item()
        recent_loss += loss.item()
        if step % 500 == 0 or step == args.steps:
            steps_in_window = step % 500 or 500
            checkpoint = {"step": step, "mean_training_log_loss": recent_loss / steps_in_window, "validation": evaluate(model, validation_tasks)}
            checkpoints.append(checkpoint)
            print(f"step={step} train={checkpoint['mean_training_log_loss']:.5f} val={checkpoint['validation']['observed_log_loss']:.5f} excess={checkpoint['validation']['expected_excess_log_loss']:.5f}", flush=True)
            recent_loss = 0.0
    model.eval()
    test = evaluate(model, sample_tasks(args.test_tasks, test_generator))
    traces = [export_trace(model, [1, 0, 0, 1], "A-pattern context"), export_trace(model, [0, 1, 1, 0], "B-pattern context")]
    parameters = {name: tensor.detach().tolist() for name, tensor in model.state_dict().items()}
    state_digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for trace in traces:
        trace["state_dict_sha256"] = state_digest
    result = {
        "metadata": {
            "name": "Mini-PFN: two bear-survival worlds",
            "is_tabpfn": False, "purpose": "A real trained educational model with a deliberately tiny known task prior.",
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "device": "cpu", "threads": 2, "deterministic_algorithms": True,
            "seed": args.seed, "training_rng_seed": args.seed + 1,
            "validation_rng_seed": args.seed + 2, "test_rng_seed": args.seed + 3,
            "steps": args.steps, "batch_size": args.batch_size, "training_tasks": args.steps * args.batch_size,
            "learning_rate": args.learning_rate, "optimizer": "Adam", "n_context": 4,
            "width": model.width, "attention_heads": 1,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "state_dict_sha256": state_digest, "training_status": "completed",
            "training_curve_metric": "Observed log loss on the same 2,048 independent validation tasks at each checkpoint.",
            "validation_tasks": 2048, "test_tasks": args.test_tasks,
            "selection": "Fixed training schedule; final weights used. No selection against test results.",
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "prior": "One world A or B drawn with equal probability per task. Independent balanced binary bear/strategy features. World A survival probability logistic(1) when B=L and logistic(-1) otherwise; world B reverses these. Outcomes independently sampled conditional on the shared world and row features.",
            "inputs": "Raw 0/1 context columns B,L,Y; raw 0/1 query B,L. No world identity, match indicator, posterior, or analytic Bayes output enters the model or its loss target.",
            "training_target": "One independently sampled Bernoulli query outcome per task; mean binary cross entropy.",
            "architecture": "Context MLP 3→32→32 and query MLP 2→32→32 with ReLU; separate linear Q,K,V maps; one scaled-dot-product cross-attention head; concatenate query representation and weighted value mixture; decoder MLP 64→32→1 with ReLU, then sigmoid.",
            "limitations": [
                "This is not TabPFN and has not been evaluated on general tabular datasets.",
                "It learns a deliberately restricted two-world prior with exactly four context rows and two binary features; it has no irrelevant-feature experiment.",
                "Attention weights describe this trained computation, not feature importance, causal effects, or posterior world probabilities.",
                "The analytic Bayesian comparison is exact for the specified prior; the neural approximation can differ.",
                "Displayed context examples are selected illustrations, while aggregate performance uses an independent set of sampled tasks.",
            ],
        },
        "training": [{"step": c["step"], "loss": c["validation"]["observed_log_loss"], "mean_training_loss": c.get("mean_training_log_loss")} for c in checkpoints],
        "checkpoints": checkpoints, "first_update": first_update,
        "evaluation": {
            "model_log_loss": test["observed_log_loss"],
            "bayes_log_loss": test["exact_bayes_observed_log_loss"],
            "ignore_context_log_loss": test["ignore_context_log_loss"],
            "n_tasks": args.test_tasks,
            **test,
        },
        "examples": {"A": traces[0], "B": traces[1]},
        "parameters": parameters,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    args.web_output.parent.mkdir(parents=True, exist_ok=True)
    args.web_output.write_text("// Generated by examples/tabular-foundation-models/mini_pfn.py; this is not TabPFN.\nexport const miniPfnResults = " + text + ";\n", encoding="utf-8")
    print(json.dumps({"held_out": test, "trace_predictions": [{"name": x["name"], "learned": x["probability"], "exact_bayes": x["exact_bayes_probability"], "attention": x["attention"]} for x in traces], "seconds": result["metadata"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
