"""
Benchmark Script for Vanilla IG Attribution on Question Answering

Usage:
    python run_eval_ig_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 1000 --steps 101
"""

import time
import random
import argparse
import numpy as np
import torch
import traceback
from tqdm import tqdm
from datasets import load_dataset

from vanilla_ig import ig_qa
from fcg_gradients import get_model_tokenizer, get_baseline_embedding
from xai_metrics import (
    calculate_log_odds_qa,
    calculate_soft_comprehensiveness_qa,
    calculate_soft_sufficiency_qa,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def run_single_example(question, context, model_name, a, b, steps,
                       device, baseline, eval_base_token_emb, topk=20,
                       n_samples=10):
    res = ig_qa(
        question=question,
        context=context,
        a=a, b=b, steps=steps,
        model_name=model_name,
        device=device,
        show_special_tokens=True,
        baseline=baseline,
    )

    model               = res["model"]
    input_embed         = res["input_embed"]
    attention_mask      = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids      = res["token_type_ids"]
    attr_start          = res["attributions_start"]
    attr_end            = res["attributions_end"]
    start_idx           = res["start_idx"]
    end_idx             = res["end_idx"]
    prob_start_orig     = res["start_prob"]
    prob_end_orig       = res["end_prob"]

    log_odd_start, log_odd_end = calculate_log_odds_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, topk=topk,
    )
    soft_comp_start, soft_comp_end = calculate_soft_comprehensiveness_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )
    soft_suff_start, soft_suff_end = calculate_soft_sufficiency_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        eval_base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )

    return {
        "tokens":            res["tokens"],
        "attributions_start": res["attributions_start"],
        "attributions_end":   res["attributions_end"],
        "predicted_answer":   res["predicted_answer"],
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "time":               res["time"],
        "log_odd_start":      log_odd_start,
        "soft_comp_start":    soft_comp_start,
        "soft_suff_start":    soft_suff_start,
        "log_odd_end":        log_odd_end,
        "soft_comp_end":      soft_comp_end,
        "soft_suff_end":      soft_suff_end,
    }


def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model         : {args.model_name}")
    print(f"Dataset       : {args.dataset}")
    print(f"Samples       : {args.num_samples}")
    print(f"IG baseline   : {args.baseline}")
    print(f"Eval baseline : {args.eval_baseline}")
    print(f"Soft samples  : {args.n_samples}")
    print(f"Top-k         : {args.topk}%")

    # Load model once to build eval_base_token_emb
    model, tokenizer = get_model_tokenizer(args.model_name, device, type="qa")
    embed = model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d)

    print(f"\nLoading dataset: {args.dataset} ...")
    dataset  = load_dataset(args.dataset, split="validation")
    data     = list(zip(dataset["question"], dataset["context"],
                        dataset["answers"], dataset["id"]))

    # Same filter as FCG QA script
    upd_data = [
        (q, c, a, i) for q, c, a, i in data
        if len((q + c).split(" ")) < 80
    ]
    print(f"Filtered samples: {len(upd_data)}")

    answerable_data = [
        {"context": item[1], "question": item[0], "answers": item[2]}
        for item in upd_data
    ]

    if len(answerable_data) > args.num_samples:
        sampled_data = random.sample(answerable_data, args.num_samples)
    else:
        sampled_data = answerable_data
        print(f"Warning: only {len(answerable_data)} samples available")

    print(f"Evaluating {len(sampled_data)} QA pairs ...\n")

    total_log_odds_start  = total_log_odds_end  = 0.0
    total_soft_comp_start = total_soft_comp_end = 0.0
    total_soft_suff_start = total_soft_suff_end = 0.0
    total_time = count = errors = 0

    for idx, example in enumerate(tqdm(sampled_data)):
        try:
            res = run_single_example(
                question=example["question"],
                context=example["context"],
                model_name=args.model_name,
                a=0, b=1,
                steps=args.steps,
                device=device,
                baseline=args.baseline,
                eval_base_token_emb=eval_base_token_emb,
                topk=args.topk,
                n_samples=args.n_samples,
            )

            total_log_odds_start  += res["log_odd_start"]
            total_log_odds_end    += res["log_odd_end"]
            total_soft_comp_start += res["soft_comp_start"]
            total_soft_comp_end   += res["soft_comp_end"]
            total_soft_suff_start += res["soft_suff_start"]
            total_soft_suff_end   += res["soft_suff_end"]
            total_time           += res["time"]
            count                += 1

            if count % args.print_step == 0:
                print(f"\n[{count}/{len(sampled_data)}] Running averages:")
                print(f"  Log-odds (start)  : {total_log_odds_start/count:.4f}")
                print(f"  Log-odds (end)    : {total_log_odds_end/count:.4f}")
                print(f"  Soft-Comp (start) : {total_soft_comp_start/count:.4f}")
                print(f"  Soft-Comp (end)   : {total_soft_comp_end/count:.4f}")
                print(f"  Soft-Suff (start) : {total_soft_suff_start/count:.4f}")
                print(f"  Soft-Suff (end)   : {total_soft_suff_end/count:.4f}")
                print(f"  Avg time          : {total_time/count:.4f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError on sample {idx}: {str(e)[:100]}")
                traceback.print_exc()
            continue

    if count > 0:
        print(f"\n{''*52}")
        print(f"Vanilla IG QA  |  {args.model_name}")
        print(f"  Log-odds (start)              : {total_log_odds_start/count:.6f}")
        print(f"  Soft-Comprehensiveness (start): {total_soft_comp_start/count:.6f}")
        print(f"  Soft-Sufficiency (start)      : {total_soft_suff_start/count:.6f}")
        print(f"  Log-odds (end)                : {total_log_odds_end/count:.6f}")
        print(f"  Soft-Comprehensiveness (end)  : {total_soft_comp_end/count:.6f}")
        print(f"  Soft-Sufficiency (end)        : {total_soft_suff_end/count:.6f}")
        print(f"  Avg time/sample               : {total_time/count:.4f}s")
        print(f"  Evaluated                     : {count}  |  Errors: {errors}")
        print(f"{''*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Vanilla IG Attribution for Question Answering"
    )
    parser.add_argument("--model_name",    type=str,
                        default="deepset/bert-base-cased-squad2")
    parser.add_argument("--dataset",       type=str, default="squad")
    parser.add_argument("--steps",         type=int, default=101)
    parser.add_argument("--num_samples",   type=int, default=1000)
    parser.add_argument("--topk",          type=int, default=50)
    parser.add_argument("--print_step",    type=int, default=100)
    parser.add_argument("--baseline",      type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding for IG integration path")
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    parser.add_argument("--n-samples",     type=int, default=10,
                        help="Stochastic samples for soft metrics")
    args = parser.parse_args()
    run_benchmark(args)