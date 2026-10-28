"""Reproduce the explicitly synthetic t-SNE asset used by the companion lesson.

This is an authoring utility, not a student software requirement. The source
vectors are analytic examples, not embeddings learned from language. Run with
an environment containing NumPy and scikit-learn; no data download is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


NEIGHBORS = 3
PARAMETERS = {
    "n_components": 2,
    "perplexity": 8.0,
    "early_exaggeration": 12.0,
    "learning_rate": 10.0,
    "max_iter": 3000,
    "n_iter_without_progress": 300,
    "min_grad_norm": 1e-7,
    "metric": "euclidean",
    "method": "exact",
    "n_jobs": 1,
}


def make_source_vectors():
    """Three nearby groups, with nonsymmetric offsets in six coordinates."""
    vectors = []
    labels = []
    for group_index, group in enumerate("ABC"):
        for member in range(6):
            angle = 2 * np.pi * member / 6 + 0.23 * group_index
            raw = np.array([
                0.15 + 1.0 * (group_index == 0),
                0.15 + 1.0 * (group_index == 1),
                0.15 + 1.0 * (group_index == 2),
                (0.17 + 0.012 * member) * np.cos(angle),
                (0.20 - 0.007 * member) * np.sin(angle),
                0.065 * np.sin(2 * angle + 0.31) + 0.012 * member,
            ], dtype=np.float64)
            # Deliberately break regular-ring ties without random sampling.
            raw[(group_index + 1) % 3] += 0.009 * member
            vectors.append(raw / np.linalg.norm(raw))
            labels.append((f"{group}{member + 1}", group))
    return np.vstack(vectors), labels


def make_initial_positions(count):
    """A fixed small spiral, independent of source group labels."""
    index = np.arange(count, dtype=np.float64)
    angle = index * np.pi * (3 - np.sqrt(5))
    radius = 1e-4 * np.sqrt((index + 1) / count)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def distances(values):
    delta = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def nearest(distance_matrix, index, labels):
    # Exclude self; label resolves any exact floating-point tie deterministically.
    return sorted(
        (j for j in range(len(labels)) if j != index),
        key=lambda j: (float(distance_matrix[index, j]), labels[j][0]),
    )[:NEIGHBORS]


def geometry_summary(distance_matrix, labels):
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i):
            target = within if labels[i][1] == labels[j][1] else between
            target.append(float(distance_matrix[i, j]))
    return {
        "withinGroup": {"min": min(within), "mean": float(np.mean(within)), "max": max(within)},
        "betweenGroups": {"min": min(between), "mean": float(np.mean(between)), "max": max(between)},
    }


def main():
    source, labels = make_source_vectors()
    initialization = make_initial_positions(len(source))
    with threadpool_limits(limits=1):
        model = TSNE(**PARAMETERS, init=initialization.copy())
        positions = model.fit_transform(source)
        repeated = TSNE(**PARAMETERS, init=initialization.copy()).fit_transform(source)
    np.testing.assert_allclose(positions, repeated, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(source, axis=1), 1, rtol=0, atol=1e-14)
    assert np.isfinite(positions).all()

    source_distance = distances(source)
    map_distance = distances(positions)
    map_span = float(np.ptp(positions, axis=0).max())
    nearest_spacing = float(np.min(map_distance[np.triu_indices(len(labels), k=1)]))
    assert nearest_spacing * 320 / map_span > 5
    source_cosine = np.clip(source @ source.T, -1, 1)
    # Unit-vector Euclidean distance and cosine imply the same exact ordering.
    np.testing.assert_allclose(source_distance**2, 2 - 2 * source_cosine,
                               rtol=0, atol=1e-14)
    points = []
    for index, (label, group) in enumerate(labels):
        source_neighbors = nearest(source_distance, index, labels)
        map_neighbors = nearest(map_distance, index, labels)
        cosine_neighbors = sorted(
            (j for j in range(len(labels)) if j != index),
            key=lambda j: (-float(source_cosine[index, j]), labels[j][0]),
        )[:NEIGHBORS]
        assert source_neighbors == cosine_neighbors
        assert all(labels[j][1] == group for j in source_neighbors)
        points.append({
            "id": label,
            "group": group,
            "vector": source[index].tolist(),
            "position": positions[index].tolist(),
            "sourceNeighbors": [labels[j][0] for j in source_neighbors],
            "mapNeighbors": [labels[j][0] for j in map_neighbors],
            "sourceNeighborDetails": [
                {"id": labels[j][0], "distance": float(source_distance[index, j]),
                 "cosine": float(source_cosine[index, j])}
                for j in source_neighbors
            ],
            "mapNeighborDetails": [
                {"id": labels[j][0], "distance": float(map_distance[index, j])}
                for j in map_neighbors
            ],
            "neighborOverlap": len(set(source_neighbors) & set(map_neighbors)),
        })

    metadata = {
        "title": "An exact t-SNE projection of 18 synthetic unit vectors",
        "synthetic": True,
        "description": "Analytically constructed six-dimensional unit vectors in three neighbor groups. These are not word embeddings and carry no claimed word semantics.",
        "sourceDimensions": 6,
        "targetDimensions": 2,
        "pointCount": len(points),
        "groups": ["A", "B", "C"],
        "neighborsCompared": NEIGHBORS,
        "sourceDistance": "Euclidean distance on unit vectors; source neighbor ordering equals decreasing cosine similarity.",
        "mapDistance": "Euclidean distance in the exported, unscaled two-dimensional t-SNE coordinates.",
        "algorithm": "scikit-learn TSNE",
        "parameters": PARAMETERS,
        "authoringChoice": {
            "description": "Perplexity 8 resolves the markers in a 320-pixel map while keeping all original point coordinates and one uniform display scale. At that perplexity, a bounded comparison selected the learning rate with the lowest final KL divergence among 5, 10, 20, and 50; every other parameter and initialization was held fixed. This yields a clear tiny example, not a general tuning recommendation.",
            "candidateLearningRatesAndKL": [
                [5.0, 0.000831383950622653],
                [10.0, 0.0008016334142905289],
                [20.0, 0.0008101321130799244],
                [50.0, 0.12183233796674349],
            ],
        },
        "geometrySummary": {
            "source": geometry_summary(source_distance, labels),
            "map": geometry_summary(map_distance, labels),
            "largestMapCoordinateSpan": map_span,
            "minimumMapPointDistance": nearest_spacing,
            "minimumPointSeparationAt320Pixels": nearest_spacing * 320 / map_span,
            "displayAssumption": "Uniform scale with the largest coordinate span filling 320 pixels, before padding; no point displacement or per-group scaling.",
        },
        "initialization": {
            "kind": "explicit analytic two-dimensional spiral",
            "formula": "theta_i = i*pi*(3-sqrt(5)); r_i = 1e-4*sqrt((i+1)/18); init_i = r_i*(cos(theta_i), sin(theta_i)); i=0,...,17",
            "positions": initialization.tolist(),
            "randomSampling": False,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
        },
        "optimization": {
            "klDivergence": float(model.kl_divergence_),
            "lastIterationIndex": int(model.n_iter_),
            "repeatMaxAbsoluteDifference": float(np.max(np.abs(positions - repeated))),
            "numericalThreadLimit": 1,
        },
        "interpretation": "Only a small synthetic demonstration. A projection can change nearest-neighbor rankings; map axes, global distances, and group spacing have no direct semantic interpretation. t-SNE does not train or replace the source vectors.",
        "sourceGenerator": "generate_projection.py",
    }
    destination = Path(__file__).resolve().parent / "projection.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "// Generated by ../generate_projection.py; explicitly synthetic data.\n"
        "// position contains raw 2D t-SNE coordinates; vector contains the original 6D unit vector.\n"
        "export const projectionMetadata = " + json.dumps(metadata, indent=2, allow_nan=False) + ";\n\n"
        "export const points = " + json.dumps(points, indent=2, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(destination),
        "points": len(points),
        "klDivergence": float(model.kl_divergence_),
        "neighborOverlapsOutOf3": {p["id"]: p["neighborOverlap"] for p in points},
        "positionRange": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "repeatMaxAbsoluteDifference": metadata["optimization"]["repeatMaxAbsoluteDifference"],
    }, indent=2))


if __name__ == "__main__":
    main()
