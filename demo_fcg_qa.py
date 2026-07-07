"""
demo_fcg_qa.py

Visualize FCGGrad vs Integrated Gradients attributions on a single
user-provided question-answering example.

Usage:
  python demo_fcg_qa.py --question "What is the capital of France?" --context "France is a country in Europe. Its capital is Paris."
  python demo_fcg_qa.py --question "Who wrote Hamlet?" --context "William Shakespeare wrote many plays including Hamlet." --baseline zero
  python demo_fcg_qa.py --question "When was the Eiffel Tower built?" --context "The Eiffel Tower was completed in 1889." --steps 200
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fcg_gradients import fcg_gradient_qa, get_model_tokenizer
from vanilla_ig import ig_qa

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


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


def _extract_viz_tokens(tokenizer, question, context, attr_s_full, attr_e_full,
                         start_idx, end_idx):
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


def visualize_qa_comparison(fcg_data, ig_data, question, filename):
    """
    Side-by-side comparison of FCGGrad and IG for QA.
    fcg_data / ig_data: (tokens, attr_s, attr_e, start_idx, end_idx, predicted_answer)
    """
    tokens, attr_s_fcg, attr_e_fcg, s_fcg, e_fcg, pred_fcg = fcg_data
    _,      attr_s_ig,  attr_e_ig,  s_ig,  e_ig,  pred_ig  = ig_data

    ns_fcg, _ = _norm_scores(attr_s_fcg)
    ne_fcg, _ = _norm_scores(attr_e_fcg)
    ns_ig,  _ = _norm_scores(attr_s_ig)
    ne_ig,  _ = _norm_scores(attr_e_ig)

    name_width  = 0.06
    label_width = 0.07
    token_font  = 7
    number_font = 5.5
    cell_pad_x  = 0.003
    cell_gap    = 0.004
    num_gap     = 0.015
    line_drop   = 0.08
    method_gap  = 0.03
    row_gap     = 0.025

    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.subplots_adjust(left=0.01, right=1.0, top=0.93, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    T = ax.transAxes
    renderer = fig.canvas.get_renderer()

    y_meta = 0.98
    meta_lines = [
        f"Q: {question}",
        f"Pred (FCGGrad): <<{pred_fcg}>>    Pred (IG): <<{pred_ig}>>",
    ]
    for line in meta_lines:
        ax.text(0.01, y_meta, line, fontsize=9, fontfamily="monospace",
                verticalalignment="top", transform=T)
        y_meta -= 0.025

    y_pos = y_meta - 0.02

    def draw_method(y_start, method_name, attr_s, attr_e, ns_arr, ne_arr,
                    s_idx, e_idx):
        y = y_start

        ax.text(name_width / 2, y, method_name, fontsize=9, fontweight="bold",
                fontfamily="monospace", verticalalignment="top",
                horizontalalignment="center", transform=T)

        x_start = name_width + 0.01

        for row_label, raw_scores, norm_vals in [
            ("Start", attr_s, ns_arr),
            ("End",   attr_e, ne_arr),
        ]:
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

            y = line_y - 0.04
            if row_label == "Start":
                y -= row_gap

        return y - 0.03

    y_pos = draw_method(y_pos, "FCGGrad", attr_s_fcg, attr_e_fcg, ns_fcg, ne_fcg,
                        s_fcg, e_fcg)
    y_pos -= method_gap
    y_pos = draw_method(y_pos, "IG", attr_s_ig, attr_e_ig, ns_ig, ne_ig,
                        s_ig, e_ig)

    fig.suptitle(
        f"FCGGrad vs IG -- QA  |  Predicted: <<{pred_fcg}>>  |  Q: {question[:80]}{'...' if len(question) > 80 else ''}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FCGGrad vs IG on a user-provided QA example"
    )
    parser.add_argument("--question", type=str, required=True,
                        help="The question")
    parser.add_argument("--context",  type=str, required=True,
                        help="The context/passage containing the answer")
    parser.add_argument("--model_name", type=str,
                        default="deepset/bert-base-cased-squad2")
    parser.add_argument("--steps",    type=int, default=100)
    parser.add_argument("--baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    parser.add_argument("--output",   type=str, default="demo_qa.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model    : {args.model_name}")
    print(f"Question : {args.question}")
    print(f"Context  : {args.context}")
    print(f"Baseline : {args.baseline}")
    print(f"Steps    : {args.steps}")
    print()

    model, tokenizer = get_model_tokenizer(args.model_name, device, type="qa")

    print("Running FCGGrad QA ...")
    res_fcg = fcg_gradient_qa(
        question=args.question,
        context=args.context,
        a=0.0, b=1.0, steps=args.steps,
        model_name=args.model_name,
        device=device,
        show_special_tokens=True,
        baseline=args.baseline,
    )

    print("Running Integrated Gradients QA ...")
    res_ig = ig_qa(
        question=args.question,
        context=args.context,
        a=0.0, b=1.0, steps=args.steps,
        model_name=args.model_name,
        device=device,
        show_special_tokens=True,
        baseline=args.baseline,
    )

    print(f"FCGGrad predicted answer : {res_fcg['predicted_answer']}")
    print(f"IG predicted answer      : {res_ig['predicted_answer']}")
    print()

    fcg_viz = _extract_viz_tokens(
        tokenizer, args.question, args.context,
        res_fcg["attributions_start"].detach().cpu().numpy(),
        res_fcg["attributions_end"].detach().cpu().numpy(),
        res_fcg["start_idx"], res_fcg["end_idx"],
    )
    fcg_viz = fcg_viz + (res_fcg["predicted_answer"],)

    ig_viz = _extract_viz_tokens(
        tokenizer, args.question, args.context,
        res_ig["attributions_start"].detach().cpu().numpy(),
        res_ig["attributions_end"].detach().cpu().numpy(),
        res_ig["start_idx"], res_ig["end_idx"],
    )
    ig_viz = ig_viz + (res_ig["predicted_answer"],)

    visualize_qa_comparison(fcg_viz, ig_viz, args.question, args.output)


if __name__ == "__main__":
    main()
