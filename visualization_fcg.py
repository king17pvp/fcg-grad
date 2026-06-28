"""
visualization_pg.py

Visualize FCGGrad attributions on sentiment classification examples.
Samples good examples from SST-2 / IMDB / Rotten Tomatoes and generates
token-level heatmap visualizations.

Usage:
  python visualization_pg.py --model distilbert --dataset sst2 --num_samples 50
  python visualization_pg.py --model bert --dataset imdb --baseline zero
  python visualization_pg.py --model roberta --dataset rotten --filter_by soft_comp --min_score 0.5
"""

import os
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fcg_gradients import fcg_gradient_classification
from vanilla_ig import ig_classification

random.seed(42)
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

LABEL_NAMES = {
    "sst2":   {0: "Negative", 1: "Positive"},
    "imdb":   {0: "Negative", 1: "Positive"},
    "rotten": {0: "Negative", 1: "Positive"},
}


def _token_color(score, norm_val):
    """Return (facecolor, edgecolor, text_color) for a token given its raw and normalized score."""
    if norm_val > 0:
        facecolor = (0.0, 0.7, 0.0, abs(norm_val) * 0.85 + 0.15)
        edgecolor = (0.0, 0.4, 0.0, 1.0)
    else:
        facecolor = (1.0, 0.0, 0.0, abs(norm_val) * 0.85 + 0.15)
        edgecolor = (0.5, 0.0, 0.0, 1.0)
    text_color = "white" if abs(norm_val) > 0.55 else "black"
    return facecolor, edgecolor, text_color


def _measure_text(ax, x, y, text, fontsize, renderer):
    """Return (width, height) of text in data coordinates."""
    t = ax.text(x, y, text, fontsize=fontsize, fontfamily="monospace",
                verticalalignment="center", alpha=0)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)
    return bbox_data[1, 0] - bbox_data[0, 0], bbox_data[1, 1] - bbox_data[0, 1]


def visualize_single(tokens, scores, predicted_label, metrics, filename,
                     title=None, max_tokens_per_line=16):
    """
    Save a single-method token attribution visualization.

    Each token is rendered as:
      ┌┐
      │  token   │  ← coloured box (green + / red −)
      └┘
       +0.0341      ← raw attribution number below the box

    Normalization uses the 95th percentile of |scores|.
    """
    scores = np.array(scores, dtype=np.float32)

    #  Normalise with a sentence-aware ceiling 
    abs_scores = np.abs(scores)
    cap = np.percentile(abs_scores, 95) if len(abs_scores) > 1 else abs_scores[0]
    if cap < 1e-10:
        cap = 1.0
    norm_scores = np.clip(scores / cap, -1.0, 1.0)

    label_name = "Positive" if predicted_label == 1 else "Negative"

    #  Figure 
    token_font  = 9
    number_font = 6.5
    cell_pad_x  = 0.006
    cell_gap    = 0.005
    num_gap     = 0.02   # vertical space between box and number
    line_h      = 0.72   # vertical space per token row (box + number)

    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()

    #  Title bar 
    if title is None:
        title = (
            f"FCGGrad — Pred: {label_name}  |  "
            f"Log-Odds: {metrics.get('log_odds', 0):+.3f}  "
            f"Soft-Comp: {metrics.get('soft_comp', 0):.3f}  "
            f"Soft-Suff: {metrics.get('soft_suff', 0):.3f}"
        )
    ax.text(0.01, 0.93, title, fontsize=11, fontweight="bold",
            transform=ax.transAxes, fontfamily="monospace")

    #  Layout tokens 
    x, y = 0.015, 0.62
    tokens_in_line = 0

    for raw_score, norm_val, token in zip(scores, norm_scores, tokens):
        facecolor, edgecolor, text_color = _token_color(raw_score, norm_val)
        num_str = f"{raw_score:+.4f}"

        # Measure token to size the coloured box
        tok_w, tok_h = _measure_text(ax, x, y, token, token_font, renderer)
        box_w = tok_w + cell_pad_x * 2
        box_h = tok_h + 0.012

        # Measure number to check cell width
        num_w, num_h = _measure_text(ax, x, y, num_str, number_font, renderer)
        cell_w = max(box_w, num_w)

        # Line wrap — use cell_w for advance
        if x + cell_w > 0.96 or tokens_in_line >= max_tokens_per_line:
            x = 0.015
            y -= line_h
            tokens_in_line = 0

        #  Coloured box (token only) 
        rect = patches.Rectangle(
            (x + (cell_w - box_w) / 2, y - box_h / 2),
            box_w, box_h,
            linewidth=0.5, edgecolor=edgecolor, facecolor=facecolor,
        )
        ax.add_patch(rect)

        # Token text centred in the box
        ax.text(x + cell_w / 2, y,
                token, fontsize=token_font, fontfamily="monospace",
                verticalalignment="center", horizontalalignment="center",
                color=text_color)

        #  Number below the box 
        ax.text(x + cell_w / 2, y - box_h / 2 - num_gap,
                num_str, fontsize=number_font, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="center",
                color="black", alpha=0.75)

        x += cell_w + cell_gap
        tokens_in_line += 1

    # Colour bar
    cbar_ax = fig.add_axes([0.92, 0.28, 0.012, 0.40])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pg", [(1, 0, 0), (1, 1, 1), (0, 0.7, 0)]
    )
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-cap, vmax=cap),
        cmap=cmap,
    )
    fig.colorbar(sm, cax=cbar_ax, label="Attribution (raw)")

    ax.set_xlim(0, 0.91)
    ax.set_ylim(y - 0.75, 0.95)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {filename}")


def visualize_comparison(methods_data, pred_label_name, filename, metrics_dict=None):
    """
    Side-by-side comparison of multiple attribution methods in a SINGLE axes
    so that the gap between methods is minimal.

    Layout:
      [PG Name]  [token] [token] ...
                 +0.03  -0.01  ...
                 Log-Odds: -2.3  Soft-Comp: 0.4  Soft-Suff: -0.1
      [IG Name]  [token] [token] ...        ← small gap from PG metrics
                 +0.02  -0.01  ...
                 Log-Odds: -1.9  Soft-Comp: 0.3  Soft-Suff: -0.1

    Args:
        methods_data: [(tokens, scores, method_name), ...]
        true_label:   ground truth label (int)
        filename:     output PNG path
        metrics_dict: {method_name: {"log_odds": ..., "soft_comp": ..., "soft_suff": ...}}
    """
    name_width   = 0.15
    token_font   = 8
    number_font  = 6
    metric_font  = 7
    cell_pad_x   = 0.005
    cell_gap     = 0.005
    num_gap      = 0.02
    line_drop    = 0.09
    method_gap   = 0.06   # small gap between method blocks

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.subplots_adjust(left=0.02, right=1.0, top=0.92, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    renderer = fig.canvas.get_renderer()

    # All positions in axes coords (0–1).  transAxes applied to every draw.
    T = ax.transAxes

    y_pos = 0.95

    for tokens, scores, method_name in methods_data:
        scores = np.array(scores, dtype=np.float32)

        abs_scores = np.abs(scores)
        cap = np.percentile(abs_scores, 95) if len(abs_scores) > 1 else max(abs_scores[0], 1e-10)
        if cap < 1e-10:
            cap = 1.0
        norm_scores = np.clip(scores / cap, -1.0, 1.0)

        # Method name
        y_tok = y_pos
        ax.text(name_width / 2, y_tok, method_name, fontsize=10,
                fontweight="bold", fontfamily="monospace",
                verticalalignment="center", horizontalalignment="center",
                transform=T)

        # Token row 
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

            # Coloured box (axes coords)
            rect = patches.Rectangle(
                (x + (cell_w - box_w) / 2, y_tok - box_h / 2),
                box_w, box_h,
                linewidth=0.5, edgecolor=edgecolor, facecolor=facecolor,
                transform=T,
            )
            ax.add_patch(rect)

            # Token text
            ax.text(x + cell_w / 2, y_tok, token,
                    fontsize=token_font, fontfamily="monospace",
                    verticalalignment="center", horizontalalignment="center",
                    color=text_color, transform=T)

            # Number below box
            ax.text(x + cell_w / 2, y_tok - box_h / 2 - num_gap,
                    num_str, fontsize=number_font, fontfamily="monospace",
                    verticalalignment="top", horizontalalignment="center",
                    color="black", alpha=0.75, transform=T)

            x += cell_w + cell_gap
            tokens_in_line += 1

        # Metrics (removed, but spacing preserved)
        y_metrics = y_tok - last_num_h - num_gap - 0.04

        # Next method
        y_pos = y_metrics - method_gap

    fig.suptitle(f"FCGGrad vs IG — Predicted label: {pred_label_name}",
                 fontsize=13, fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    print(f"  Saved: {filename}")


# Metric computation on the fly

def compute_metrics_simple(res, eval_model, tokenizer, device, eval_baseline, n_samples):
    """Compute log-odds, soft-comp, soft-suff for a single FCG result."""
    from fcg_gradients import get_baseline_embedding
    from xai_metrics import (
        calculate_log_odds,
        calculate_soft_comprehensiveness,
        calculate_soft_sufficiency,
    )
    from helpers.distilbert_helper import nn_forward_func as d_nn
    from helpers.bert_helper import nn_forward_func as b_nn
    from helpers.roberta_helper import nn_forward_func as r_nn

    embed = eval_model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X = embed(dummy_ids)
    eval_base_token_emb = get_baseline_embedding(
        eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]

    if "distilbert" in res.get("model_name", ""):
        nn_forward_func = d_nn
    elif "roberta" in res.get("model_name", ""):
        nn_forward_func = r_nn
    else:
        nn_forward_func = b_nn

    log_odd, _ = calculate_log_odds(
        nn_forward_func, eval_model,
        res["input_embed"], res["position_embed"], res["type_embed"],
        res["attention_mask"], eval_base_token_emb,
        res["attr_full"], topk=20,
    )
    comp = calculate_soft_comprehensiveness(
        nn_forward_func, eval_model,
        res["input_embed"], res["position_embed"], res["type_embed"],
        res["attention_mask"], eval_base_token_emb,
        res["attr_full"], n_samples=n_samples,
    )
    suff = calculate_soft_sufficiency(
        nn_forward_func, eval_model,
        res["input_embed"], res["position_embed"], res["type_embed"],
        res["attention_mask"], eval_base_token_emb,
        res["attr_full"], n_samples=n_samples,
    )
    return {"log_odds": log_odd, "soft_comp": comp, "soft_suff": suff}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FCGGrad attributions on sentiment data"
    )
    parser.add_argument("--model",       type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",     type=str, default="sst2",
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",       type=int, default=100)
    parser.add_argument("--baseline",    type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--n-samples",   type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Max examples to visualize")
    parser.add_argument("--min_score",   type=float, default=3.0,
                        help="Min margin: FCG log_odds must be < IG log_odds by this much")
    parser.add_argument("--output_dir",  type=str, default="visualizations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[(args.model, args.dataset)]

    # Auto-set min_score thresholds
    if args.min_score is None:
        args.min_score = 3.0   # FCG log_odds must be at least this much lower than IG

    # Filter: keep if FCG log_odds < IG log_odds - min_score
    # (more negative = better, FCG must beat IG by min_score margin)

    print(f"Model      : {model_name}")
    print(f"Dataset    : {args.dataset}")
    print(f"Baseline   : {args.baseline}")
    print(f"Filter     : FCG log_odds < IG log_odds − {args.min_score}")
    print(f"Max images : {args.num_samples}")
    print(f"Output dir : {args.output_dir}")

    # Load model for metric computation
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model.eval()

    # Load dataset
    if args.dataset == "imdb":
        dataset = load_dataset("imdb")["test"]
        data = list(zip(dataset["text"], dataset["label"]))
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data = list(zip(dataset["sentence"], dataset["label"]))
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(dataset["text"], dataset["label"]))

    print(f"\nScanning {len(data)} examples for good visualizations ...\n")

    saved = 0
    skipped = 0

    for idx, (text, true_label) in enumerate(tqdm(data)):
        if saved >= args.num_samples:
            break

        # Run PG
        try:
            res_pg = fcg_gradient_classification(
                sentence=text, a=0.0, b=1.0, steps=args.steps,
                model_name=model_name,
                show_special_tokens=False,
                baseline=args.baseline,
                device=device,
            )
        except Exception:
            skipped += 1
            continue

        # Run IG 
        try:
            res_ig = ig_classification(
                sentence=text, a=0.0, b=1.0, steps=args.steps,
                model_name=model_name,
                show_special_tokens=False,
                baseline=args.baseline,
                device=device,
            )
        except Exception:
            skipped += 1
            continue

        res_pg["model_name"] = model_name
        res_ig["model_name"] = model_name

        # Compute metrics for PG (used for filtering)
        metrics_pg = compute_metrics_simple(
            res_pg, eval_model, tokenizer, device,
            args.eval_baseline, args.n_samples,
        )
        metrics_ig = compute_metrics_simple(
            res_ig, eval_model, tokenizer, device,
            args.eval_baseline, args.n_samples,
        )

        # Filter: keep only if FCG log_odds < IG log_odds - min_score
        if not (metrics_pg["log_odds"] < metrics_ig["log_odds"] - args.min_score):
            skipped += 1
            continue

        pred = res_pg["predicted_label"]
        label_name = LABEL_NAMES[args.dataset].get(true_label, str(true_label))
        pred_name = LABEL_NAMES[args.dataset].get(pred, str(pred))

        # Comparison visualization: PG (top) + IG (bottom)
        filename = os.path.join(
            args.output_dir,
            f"fcggrad_vs_ig_{args.model}_{args.dataset}_{idx}_label{true_label}_pred{pred}.png",
        )
        visualize_comparison(
            [
                (res_pg["tokens"], res_pg["attributions"].numpy(), "FCGGrad"),
                (res_ig["tokens"], res_ig["attributions"].numpy(), "Integrated Gradients"),
            ],
            pred_label_name=pred_name,
            filename=filename,
            metrics_dict={"FCGGrad": metrics_pg, "Integrated Gradients": metrics_ig},
        )
        saved += 1

    print(f"\nDone. Saved {saved} visualizations to {args.output_dir}/")
    print(f"Skipped: {skipped} (filtered or errored)")


if __name__ == "__main__":
    main()
