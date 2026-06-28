"""
visualization_pg_qa.py

Side-by-side comparison of FCGGrad (FCGGrad) and IG attributions on QA.
Shows start + end position attributions with predicted answer spans highlighted.

Usage:
  python visualization_pg_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 30
  python visualization_pg_qa.py --baseline zero --filter_by soft_comp --min_score 0.3
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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from fcg_gradients import fcg_gradient_qa, get_model_tokenizer, get_baseline_embedding
from vanilla_ig import ig_qa
from xai_metrics import (
    calculate_log_odds_qa,
    calculate_soft_comprehensiveness_qa,
    calculate_soft_sufficiency_qa,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


#  Helpers 

def _norm_scores(scores):
    a = np.array(scores, dtype=np.float32)
    abs_a = np.abs(a)
    cap = np.percentile(abs_a, 95) if len(abs_a) > 1 else max(abs_a[0], 1e-10)
    if cap < 1e-10:
        cap = 1.0
    return np.clip(a / cap, -1.0, 1.0), cap


def _tok_color(norm_val):
    if norm_val > 0:
        fc = (0.0, 0.7, 0.0, abs(norm_val) * 0.85 + 0.15)
        ec = (0.0, 0.4, 0.0, 1.0)
    else:
        fc = (1.0, 0.0, 0.0, abs(norm_val) * 0.85 + 0.15)
        ec = (0.5, 0.0, 0.0, 1.0)
    tc = "white" if abs(norm_val) > 0.55 else "black"
    return fc, ec, tc


def _measure(ax, x, y, text, fs, renderer):
    t = ax.text(x, y, text, fontsize=fs, fontfamily="monospace",
                verticalalignment="center", alpha=0)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    inv = ax.transData.inverted()
    bd = inv.transform(bbox)
    return bd[1, 0] - bd[0, 0], bd[1, 1] - bd[0, 1]


def compute_qa_metrics(res, eval_base_token_emb, topk=50, n_samples=10):
    model = res["model"]
    input_embed = res["input_embed"]
    attention_mask = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids = res["token_type_ids"]
    attr_start = res["attributions_start"]
    attr_end = res["attributions_end"]
    start_idx = res["start_idx"]
    end_idx = res["end_idx"]
    prob_start_orig = res["start_prob"]
    prob_end_orig = res["end_prob"]

    lo_s, lo_e = calculate_log_odds_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, topk=topk,
    )
    sc_s, sc_e = calculate_soft_comprehensiveness_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )
    ss_s, ss_e = calculate_soft_sufficiency_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )
    return {
        "log_odds_start": lo_s, "log_odds_end": lo_e,
        "soft_comp_start": sc_s, "soft_comp_end": sc_e,
        "soft_suff_start": ss_s, "soft_suff_end": ss_e,
    }


def _extract_viz_tokens(tokenizer, question, context, attr_s_full, attr_e_full,
                         start_idx, end_idx):
    """Return (tokens, attr_s, attr_e, filt_start, filt_end) without special tokens."""
    enc = tokenizer(question, context, return_special_tokens_mask=True)
    all_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    sp_mask = enc["special_tokens_mask"]

    viz_tokens, viz_s, viz_e = [], [], []
    full_to_filt = {}
    fi = 0
    for i, (tok, is_sp) in enumerate(zip(all_tokens, sp_mask)):
        if not is_sp and i < len(attr_s_full):
            viz_tokens.append(tok)
            viz_s.append(attr_s_full[i])
            viz_e.append(attr_e_full[i])
            full_to_filt[i] = fi
            fi += 1

    filt_start = full_to_filt.get(start_idx, 0)
    filt_end = full_to_filt.get(end_idx, len(viz_tokens) - 1)
    return viz_tokens, np.array(viz_s), np.array(viz_e), filt_start, filt_end


#  Visualization 

def visualize_qa_comparison(pg_data, ig_data, question, gold_answers,
                             metrics_pg, metrics_ig, filename):
    """
    Compare FCGGrad (top) and IG (bottom) on the same QA example.
    Each method shows Start + End attribution rows.

    pg_data / ig_data: (tokens, attr_s, attr_e, start_idx, end_idx, predicted_answer)
    """
    tokens, attr_s_pg, attr_e_pg, s_pg, e_pg, pred_pg = pg_data
    _,      attr_s_ig, attr_e_ig, s_ig, e_ig, pred_ig = ig_data

    ns_pg, cap_s_pg = _norm_scores(attr_s_pg)
    ne_pg, cap_e_pg = _norm_scores(attr_e_pg)
    ns_ig, cap_s_ig = _norm_scores(attr_s_ig)
    ne_ig, cap_e_ig = _norm_scores(attr_e_ig)

    name_width  = 0.06
    label_width = 0.07
    token_font  = 7
    number_font = 5.5
    cell_pad_x  = 0.003
    cell_gap    = 0.004
    num_gap     = 0.015
    line_drop   = 0.08
    method_gap  = 0.03
    row_gap     = 0.025  # gap between start and end rows within a method

    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.subplots_adjust(left=0.01, right=1.0, top=0.93, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    T = ax.transAxes
    renderer = fig.canvas.get_renderer()

    #  Metadata 
    gold_str = " | ".join(gold_answers[:3]) if gold_answers else "(none)"
    y_meta = 0.98
    meta_lines = [
        f"Q: {question}",
        f"Pred (FCGGrad): «{pred_pg}»    Pred (IG): «{pred_ig}»    Gold: «{gold_str}»",
    ]
    for line in meta_lines:
        ax.text(0.01, y_meta, line, fontsize=9, fontfamily="monospace",
                verticalalignment="top", transform=T)
        y_meta -= 0.025

    y_pos = y_meta - 0.02

    #  Draw one method block 
    def draw_method(y_start, method_name, attr_s, attr_e, ns_arr, ne_arr,
                    s_idx, e_idx, metrics):
        """Returns the y coordinate just below the metrics line."""
        y = y_start

        # Method name
        ax.text(name_width / 2, y, method_name, fontsize=9, fontweight="bold",
                fontfamily="monospace", verticalalignment="top",
                horizontalalignment="center", transform=T)

        x_start = name_width + 0.01

        # Shared token row (labels on left for each row)
        for row_label, raw_scores, norm_vals in [
            ("Start", attr_s, ns_arr),
            ("End",   attr_e, ne_arr),
        ]:
            # Row label
            ax.text(name_width + label_width / 2, y, row_label, fontsize=7,
                    fontfamily="monospace", verticalalignment="top",
                    horizontalalignment="center", color="grey", transform=T)

            x = x_start + label_width
            line_y = y
            tokens_in_line = 0
            max_per_line = 18

            for i, (token, raw, nv) in enumerate(zip(tokens, raw_scores, norm_vals)):
                in_span = s_idx <= i <= e_idx
                fc, ec, tc = _tok_color(nv)
                num_str = f"{raw:+.4f}"

                tok_w, tok_h = _measure(ax, x, line_y, token, token_font, renderer)
                box_w = tok_w + cell_pad_x * 2
                box_h = tok_h + 0.006

                num_w, num_h = _measure(ax, x, line_y, num_str, number_font, renderer)
                cell_w = max(box_w, num_w)

                if x + cell_w > 0.98 or tokens_in_line >= max_per_line:
                    x = x_start + label_width
                    line_y -= line_drop
                    tokens_in_line = 0

                lw = 2.0 if in_span else 0.5
                rect = patches.Rectangle(
                    (x + (cell_w - box_w) / 2, line_y - box_h / 2),
                    box_w, box_h,
                    linewidth=lw,
                    edgecolor=(0.0, 0.0, 0.6, 1.0) if in_span else ec,
                    facecolor=fc, transform=T,
                )
                ax.add_patch(rect)

                ax.text(x + cell_w / 2, line_y, token,
                        fontsize=token_font, fontfamily="monospace",
                        verticalalignment="center", horizontalalignment="center",
                        color=tc, transform=T)

                ax.text(x + cell_w / 2, line_y - box_h / 2 - num_gap,
                        num_str, fontsize=number_font, fontfamily="monospace",
                        verticalalignment="top", horizontalalignment="center",
                        color="black", alpha=0.75, transform=T)

                x += cell_w + cell_gap
                tokens_in_line += 1

            y = line_y - 0.04  # gap after numbers, before next row
            if row_label == "Start":
                y -= row_gap

        # Metrics (removed, spacing preserved)
        return y - 0.03

    y_pos = draw_method(y_pos, "FCGGrad", attr_s_pg, attr_e_pg, ns_pg, ne_pg,
                        s_pg, e_pg, metrics_pg)
    y_pos -= method_gap
    y_pos = draw_method(y_pos, "IG", attr_s_ig, attr_e_ig, ns_ig, ne_ig,
                        s_ig, e_ig, metrics_ig)

    fig.suptitle(
        f"FCGGrad vs IG — QA  |  Predicted: «{pred_pg}»  |  Q: {question[:80]}{'…' if len(question) > 80 else ''}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    print(f"  Saved: {filename}")


#  Main 

def main():
    parser = argparse.ArgumentParser(
        description="Visualize FCGGrad vs IG QA attributions on SQuAD"
    )
    parser.add_argument("--model_name",    type=str,
                        default="deepset/bert-base-cased-squad2")
    parser.add_argument("--dataset",       type=str, default="squad")
    parser.add_argument("--steps",         type=int, default=100)
    parser.add_argument("--baseline",      type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--topk",          type=int, default=50)
    parser.add_argument("--n-samples",     type=int, default=10)
    parser.add_argument("--num_samples",   type=int, default=30)
    parser.add_argument("--min_score",     type=float, default=2.0,
                        help="Min margin: FCG avg log_odds < IG avg log_odds by this much")
    parser.add_argument("--output_dir",    type=str, default="visualizations_qa")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.min_score is None:
        args.min_score = 1.0

    # Filter: FCG avg log_odds < IG avg log_odds - min_score
    better = lambda pg_lo, ig_lo: pg_lo < ig_lo - args.min_score

    print(f"Model         : {args.model_name}")
    print(f"Dataset       : {args.dataset}")
    print(f"Baseline      : {args.baseline}")
    print(f"Eval baseline : {args.eval_baseline}")
    print(f"Filter        : FCG avg log_odds < IG avg log_odds − {args.min_score}")
    print(f"Max images    : {args.num_samples}")
    print(f"Output dir    : {args.output_dir}")

    #  Load model, build eval_base_token_emb 
    model, tokenizer = get_model_tokenizer(args.model_name, device, type="qa")
    embed = model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X = embed(dummy_ids)
    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]

    #  Load dataset 
    print(f"\nLoading dataset: {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="validation")
    data = list(zip(
        dataset["question"], dataset["context"],
        dataset["answers"], dataset["id"],
    ))
    upd_data = [(q, c, a, i) for q, c, a, i in data
                if len((q + c).split(" ")) < 80]
    print(f"Filtered samples: {len(upd_data)}")

    answerable = [{"context": c, "question": q, "answers": a}
                  for q, c, a, _ in upd_data]
    # if len(answerable) > 500:
    #     answerable = random.sample(answerable, 500)

    print(f"Scanning {len(answerable)} examples ...\n")

    saved, skipped = 0, 0
    a, b = 0, 1

    for example in tqdm(answerable):
        if saved >= args.num_samples:
            break

        question = example["question"]
        context = example["context"]
        gold_answers = example["answers"]["text"]

        #  Run FCGGrad 
        try:
            res_pg = fcg_gradient_qa(
                question=question, context=context,
                a=a, b=b, steps=args.steps,
                model_name=args.model_name,
                device=device, show_special_tokens=True,
                baseline=args.baseline,
            )
        except Exception:
            skipped += 1
            continue

        #  Run IG 
        try:
            res_ig = ig_qa(
                question=question, context=context,
                a=a, b=b, steps=args.steps,
                model_name=args.model_name,
                device=device, show_special_tokens=True,
                baseline=args.baseline,
            )
        except Exception:
            skipped += 1
            continue

        #  Metrics 
        metrics_pg = compute_qa_metrics(
            res_pg, eval_base_token_emb, topk=args.topk, n_samples=args.n_samples)
        metrics_ig = compute_qa_metrics(
            res_ig, eval_base_token_emb, topk=args.topk, n_samples=args.n_samples)

        # Filter: keep if FCG avg log_odds < IG avg log_odds - min_score
        pg_lo_start = metrics_pg["log_odds_start"]
        ig_lo_start = metrics_ig["log_odds_start"]
        pg_lo_end = metrics_pg["log_odds_end"]
        ig_lo_end = metrics_ig["log_odds_end"]
        
        if not better(pg_lo_start, ig_lo_start) or not better(pg_lo_end, ig_lo_end):
            skipped += 1
            continue
        
        #  Extract viz tokens 
        try:
            pg_viz = _extract_viz_tokens(
                tokenizer, question, context,
                res_pg["attributions_start"].detach().cpu().numpy(),
                res_pg["attributions_end"].detach().cpu().numpy(),
                res_pg["start_idx"], res_pg["end_idx"],
            )
            pg_viz = pg_viz + (res_pg["predicted_answer"],)

            ig_viz = _extract_viz_tokens(
                tokenizer, question, context,
                res_ig["attributions_start"].detach().cpu().numpy(),
                res_ig["attributions_end"].detach().cpu().numpy(),
                res_ig["start_idx"], res_ig["end_idx"],
            )
            ig_viz = ig_viz + (res_ig["predicted_answer"],)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"\n  extract error: {e}")
            continue

        filename = os.path.join(
            args.output_dir,
            f"fcggrad_vs_ig_qa_{saved:03d}.png",
        )
        try:
            visualize_qa_comparison(
                pg_viz, ig_viz, question, gold_answers,
                metrics_pg, metrics_ig, filename,
            )
            saved += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"\n  viz error: {e}")

    print(f"\nDone. Saved {saved} visualizations to {args.output_dir}/")
    print(f"Skipped: {skipped} (filtered or errored)")


if __name__ == "__main__":
    main()
