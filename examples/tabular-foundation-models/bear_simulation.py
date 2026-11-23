"""Shared bear-survival simulation with nested independent distractor columns.

The original three-column draw order is retained exactly, including the outcome
noise draw. Extra columns use a separate fixed random-number stream, so adding
distractors preserves bear types, responses, outcomes, and the first three noise
columns. Train/test splitting remains the caller's responsibility.
"""
from __future__ import annotations

import numpy as np


def simulate(seed: int, n: int, n_noise: int = 3):
    """Generate the original task and append independent Uniform(0,1) columns."""
    if not isinstance(n_noise, int) or n_noise < 0:
        raise ValueError("n_noise must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    brown = rng.integers(0, 2, n)
    lie_down = rng.integers(0, 2, n)
    original_noise = rng.uniform(size=(n, 3))
    logistic_error = rng.logistic(size=n)
    y = (-1 + 2 * (brown == lie_down) + logistic_error > 0).astype(int)
    if n_noise <= 3:
        noise = original_noise[:, :n_noise]
    else:
        extension = np.random.default_rng(np.random.SeedSequence([seed, 4728, 20261123]))
        # Draw one full column at a time, retaining prefixes as n_noise grows.
        extra_noise = extension.uniform(size=(n_noise - 3, n)).T
        noise = np.column_stack([original_noise, extra_noise])
    signal = np.column_stack([brown, lie_down])
    continuous = np.column_stack([signal, noise])
    return {
        "signal_only": signal,
        "continuous_noise": continuous,
        "binary_noise": np.column_stack([signal, noise > 0.5]),
        "interaction_feature": np.column_stack([continuous, brown * lie_down]),
        "two_bins": continuous,
    }, y
