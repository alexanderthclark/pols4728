#!/usr/bin/env python3
"""Course styling for a standard SHAP local waterfall, with an optional export CLI.

The plot is built by shap.plots.waterfall; this helper changes presentation only.
It uses public Matplotlib artist methods rather than SHAP's private style globals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.text import Text
import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENS = ROOT / "design" / "tokens.json"
DEFAULT_DATA = ROOT / "docs" / "shap" / "data.json"


def course_rc(tokens):
    """Return Matplotlib settings from the guide's print/preview tokens."""
    colors, typography, plot = tokens["colors"], tokens["type"], tokens["plot"]
    return {
        "figure.facecolor": colors["paper"],
        "axes.facecolor": colors["paper"],
        "savefig.facecolor": colors["paper"],
        "text.color": colors["ink"],
        "axes.labelcolor": colors["ink"],
        "axes.edgecolor": colors["rule"],
        "xtick.color": colors["muted"],
        "ytick.color": colors["ink"],
        "font.family": [typography["preview_text"], "DejaVu Serif"],
        "font.size": typography["plot_label_pt"],
        "mathtext.fontset": typography["preview_math"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.labelsize": typography["plot_label_pt"],
        "axes.titlesize": typography["plot_title_pt"],
        "xtick.labelsize": typography["plot_tick_pt"],
        "ytick.labelsize": typography["plot_tick_pt"],
        "axes.linewidth": plot["axis_pt"],
        "lines.linewidth": plot["line_pt"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


def _matches_color(value, expected):
    try:
        return np.allclose(to_rgba(value), to_rgba(expected), rtol=0, atol=1e-8)
    except (TypeError, ValueError):
        return False


def course_waterfall(explanation, *, title=None, output_label="Model output", tokens_path=DEFAULT_TOKENS):
    """Return the main Axes of a standard, course-styled single-row SHAP plot.

    Contributions stay in the Explanation's output units. Blue means positive
    and rust negative; signed labels and arrow direction also show that meaning.
    The context restores Matplotlib settings and interactive state on return.
    """
    tokens = json.loads(Path(tokens_path).read_text(encoding="utf-8"))
    colors, typography, plot = tokens["colors"], tokens["type"], tokens["plot"]
    positive_default = shap.plots.colors.red_rgb
    negative_default = shap.plots.colors.blue_rgb
    was_interactive = plt.isinteractive()
    try:
        with mpl.rc_context(course_rc(tokens)):
            figure = plt.figure()
            shap.plots.waterfall(explanation, max_display=len(explanation.values), show=False)
            axis = figure.axes[0]  # SHAP returns its final twin axis, not the main feature axis.
            figure.set_size_inches(plot["full_width_in"], plot["full_width_in"] * plot["height_ratio"])

            # Recolor the actual standard SHAP arrows and invisible sizing patches.
            # Their x coordinates, widths, sorting, and data are left intact.
            for plot_axis in figure.axes:
                plot_axis.set_facecolor(colors["paper"])
                for patch in plot_axis.patches:
                    for getter, setter in (
                        (patch.get_facecolor, patch.set_facecolor),
                        (patch.get_edgecolor, patch.set_edgecolor),
                    ):
                        original = getter()
                        if _matches_color(original, positive_default):
                            setter(colors["blue"])
                        elif _matches_color(original, negative_default):
                            setter(colors["rust"])
                for line in plot_axis.lines:
                    line.set_color(colors["rule"])
                    line.set_linewidth(plot["axis_pt"])
                for spine in plot_axis.spines.values():
                    spine.set_color(colors["rule"])
                    spine.set_linewidth(plot["axis_pt"])

            for text in figure.findobj(match=Text):
                text.set_fontfamily([typography["preview_text"], "DejaVu Serif"])
                text.set_fontsize(typography["plot_label_pt"])
                original = text.get_color()
                if _matches_color(original, positive_default):
                    text.set_color(colors["blue"])
                elif _matches_color(original, negative_default):
                    text.set_color(colors["rust"])
                elif _matches_color(original, "white"):
                    text.set_color(colors["paper"])
                elif _matches_color(original, "#999999"):
                    text.set_color(colors["muted"])
                else:
                    text.set_color(colors["ink"])

            axis.set_xlabel(output_label, labelpad=12)
            if title:
                figure.suptitle(title, fontsize=typography["plot_title_pt"], y=1.04)
            figure.canvas.draw()
            return axis
    finally:
        plt.interactive(was_interactive)


def export_profile(output, *, profile_id=None, data_path=DEFAULT_DATA, tokens_path=DEFAULT_TOKENS):
    """Export one saved explanation to SVG, PNG, or PDF; do not refit a model."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    profile_id = profile_id or data["defaultObservationId"]
    try:
        observation = next(row for row in data["observations"] if row["id"] == profile_id)
    except StopIteration as error:
        raise ValueError(f"Unknown profile {profile_id!r}.") from error
    explanation = shap.Explanation(
        values=np.array(observation["shapValues"], dtype=float),
        base_values=observation["baseValue"],
        data=np.array(observation["values"], dtype=float),
        feature_names=[feature["label"] for feature in data["features"]],
    )
    output = Path(output)
    if output.suffix.lower() not in {".svg", ".png", ".pdf"}:
        raise ValueError("Choose an .svg, .png, or .pdf output file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    axis = course_waterfall(
        explanation, title=observation["name"], output_label="Yearly earnings ($1,000)", tokens_path=tokens_path
    )
    try:
        tokens = json.loads(Path(tokens_path).read_text(encoding="utf-8"))
        with mpl.rc_context(course_rc(tokens)):
            axis.figure.savefig(output, bbox_inches="tight", dpi=180)
    finally:
        plt.close(axis.figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Save an SVG, PNG, or PDF outside the source files.")
    parser.add_argument("--profile", help="Profile ID from docs/shap/data.json; defaults to the story's selected profile.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    arguments = parser.parse_args()
    print(export_profile(arguments.output, profile_id=arguments.profile, data_path=arguments.data, tokens_path=arguments.tokens))
