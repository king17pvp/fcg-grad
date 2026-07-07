"""
demo_fcg_sentiment.py

Visualize FCGGrad vs Integrated Gradients attributions on a single
user-provided sentiment example.

Usage:
  python demo_fcg_sentiment.py --sentence "This movie was surprisingly great."
  python demo_fcg_sentiment.py --sentence "Terrible service, never coming back." --model bert --dataset imdb
  python demo_fcg_sentiment.py --sentence "A bit slow but the ending was worth it." --baseline zero --steps 200
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fcg_gradients import fcg_gradient_classification
from vanilla_ig import ig_classification

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

MODEL_NAMES = {
    ("distilbert", "sst2"):   "distilbert-base-uncased-finetuned-sst-2-english",
    ("distilbert", "imdb"):   "textattack/distilbert-base-uncased-imdb",
    ("distilbert", "rotten"): "textattack/distilbert-base-uncased-rotten-tomatoes",
    ("bert",       "sst2"):   "textattack/bert-base-uncased-SST-2",
    ("bert",       "imdb"):   "textattack/bert-base-uncased-imdb",
    ("bert",       "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
    ("roberta",    "sst2"):   "textattack/roberta-base-SST-2",
    ("roberta",    "imdb"):   "textattack/roberta-base-imdb",
    ("roberta",    "rotten"): "textattack/roberta-base-rotten-tomatoes",
}


def _token_color(score, norm_val):
    if norm_val > 0:
        facecolor = (0.0, 0.7, 0.0, abs(norm_val) * 0.85 + 0.15)
        edgecolor = (0.0, 0.4, 0.0, 1.0)
    else:
        facecolor = (1.0, 0.0, 0.0, abs(norm_val) * 0.85 + 0.15)
        edgecolor = (0.5, 0.0, 0.0, 1.0)
    text_color = "white" if abs(norm_val) > 0.55 else "black"
    return facecolor, edgecolor, text_color


def _measure_text(ax, x, y, text, fontsize, renderer):
    t = ax.text(x, y, text, fontsize=fontsize, fontfamily="monospace",
                verticalalignment="center", alpha=0)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)
    return bbox_data[1, 0] - bbox_data[0, 0], bbox_data[1, 1] - bbox_data[0, 1]


def visualize_comparison(methods_data, pred_label_name, filename):
    """
    Side-by-side comparison of FCGGrad and IG attributions.
    methods_data: [(tokens, scores, method_name), ...]
    """
    name_width   = 0.15
    token_font   = 8
    number_font  = 6
    cell_pad_x   = 0.005
    cell_gap     = 0.005
    num_gap      = 0.02
    line_drop    = 0.09
    method_gap   = 0.06

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.subplots_adjust(left=0.02, right=1.0, top=0.92, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    renderer = fig.canvas.get_renderer()
    T = ax.transAxes

    y_pos = 0.95

    for tokens, scores, method_name in methods_data:
        scores = np.array(scores, dtype=np.float32)

        abs_scores = np.abs(scores)
        cap = np.percentile(abs_scores, 95) if len(abs_scores) > 1 else max(abs_scores[0], 1e-10)
        if cap < 1e-10:
            cap = 1.0
        norm_scores = np.clip(scores / cap, -1.0, 1.0)

        y_tok = y_pos
        ax.text(name_width / 2, y_tok, method_name, fontsize=10,
                fontweight="bold", fontfamily="monospace",
                verticalalignment="center", horizontalalignment="center",
                transform=T)

        x_start = name_width + 0.01
        x = x_start
        tokens_in_line = 0
        max_per_line = 16
        last_num_h = 0.0

        for raw_score, norm_val, token in zip(scores, norm_scores, tokens):
            facecolor, edgecolor, text_color = _token_color(raw_score, norm_val)
            num_str = f"{raw_score:+.4f}"

            tok_w, tok_h = _measure_text(ax, x, y_tok, token, token_font, renderer)
            box_w = tok_w + cell_pad_x * 2
            box_h = tok_h + 0.006

            num_w, num_h = _measure_text(ax, x, y_tok, num_str, number_font, renderer)
            last_num_h = num_h
            cell_w = max(box_w, num_w)

            if x + cell_w > 0.98 or tokens_in_line >= max_per_line:
                x = x_start
                y_tok -= line_drop
                tokens_in_line = 0

            rect = patches.Rectangle(
                (x + (cell_w - box_w) / 2, y_tok - box_h / 2),
                box_w, box_h,
                linewidth=0.5, edgecolor=edgecolor, facecolor=facecolor,
                transform=T,
            )
            ax.add_patch(rect)

            ax.text(x + cell_w / 2, y_tok, token,
                    fontsize=token_font, fontfamily="monospace",
                    verticalalignment="center", horizontalalignment="center",
                    color=text_color, transform=T)

            ax.text(x + cell_w / 2, y_tok - box_h / 2 - num_gap,
                    num_str, fontsize=number_font, fontfamily="monospace",
                    verticalalignment="top", horizontalalignment="center",
                    color="black", alpha=0.75, transform=T)

            x += cell_w + cell_gap
            tokens_in_line += 1

        y_metrics = y_tok - last_num_h - num_gap - 0.04
        y_pos = y_metrics - method_gap

    fig.suptitle(f"FCGGrad vs IG — Predicted label: {pred_label_name}",
                 fontsize=13, fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FCGGrad vs IG on a user-provided sentiment example"
    )
    parser.add_argument("--sentence", type=str, required=True,
                        help="The sentence to analyze")
    parser.add_argument("--model",    type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",  type=str, default="sst2",
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",    type=int, default=100)
    parser.add_argument("--baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--output",   type=str, default="demo_sentiment.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[(args.model, args.dataset)]

    print(f"Model    : {model_name}")
    print(f"Sentence : {args.sentence}")
    print(f"Baseline : {args.baseline}")
    print(f"Steps    : {args.steps}")
    print()

    print("Running FCGGrad ...")
    res_fcg = fcg_gradient_classification(
        sentence=args.sentence, a=0.0, b=1.0, steps=args.steps,
        model_name=model_name,
        show_special_tokens=False,
        baseline=args.baseline,
        device=device,
    )

    print("Running Integrated Gradients ...")
    res_ig = ig_classification(
        sentence=args.sentence, a=0.0, b=1.0, steps=args.steps,
        model_name=model_name,
        show_special_tokens=False,
        baseline=args.baseline,
        device=device,
    )

    pred_label = res_fcg["predicted_label"]
    pred_name = "Positive" if pred_label == 1 else "Negative"

    print(f"Predicted label : {pred_name} ({pred_label})")
    print()

    visualize_comparison(
        [
            (res_fcg["tokens"], res_fcg["attributions"].numpy(), "FCGGrad"),
            (res_ig["tokens"], res_ig["attributions"].numpy(), "Integrated Gradients"),
        ],
        pred_label_name=pred_name,
        filename=args.output,
    )


if __name__ == "__main__":
    main()
