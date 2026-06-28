"""
Ablation Study: Effect of Reference Embedding Token on FCGGrad (PG)

Tests whether [MASK] is the right reference token for PG integration path.
Compares: mask, pad, zero, mean (of vocab), random

Runs across bert, distilbert, roberta on sst2, imdb, rotten_tomatoes.
"""

import time
from tqdm import tqdm
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from xai_metrics import *
from fcg_gradients import fcg_gradient_classification, get_baseline_embedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

BASELINES = ["mask", "pad", "zero", "mean", "random"]

BASELINE_LABEL = {
    "mask":   "[MASK]",
    "pad":    "[PAD]",
    "zero":   "zeros",
    "mean":   "μ(vocab)",
    "random": "random",
}

MODEL_CONFIGS = {
    "distilbert": {
        "sst2":   "distilbert-base-uncased-finetuned-sst-2-english",
        "imdb":   "textattack/distilbert-base-uncased-imdb",
        "rotten": "textattack/distilbert-base-uncased-rotten-tomatoes",
    },
    "bert": {
        "sst2":   "textattack/bert-base-uncased-SST-2",
        "imdb":   "textattack/bert-base-uncased-imdb",
        "rotten": "textattack/bert-base-uncased-rotten-tomatoes",
    },
    "roberta": {
        "sst2":   "textattack/roberta-base-SST-2",
        "imdb":   "textattack/roberta-base-imdb",
        "rotten": "textattack/roberta-base-rotten-tomatoes",
    },
}


def load_dataset_split(dataset_name: str):
    """Load and return (texts, labels) for a given dataset."""
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, 2000)
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


def evaluate_baseline(
    model_name: str,
    data: list,
    baseline: str,
    eval_baseline: str,
    steps: int,
    n_samples: int,
    device: str,
    a: float = 0.0,
    b: float = 1.0,
):
    """
    Run PG evaluation for a single (model, dataset, baseline) configuration.
    Returns dict of aggregated metrics.
    """
    # Load model once per config
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    embed = model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)

    eval_base_token_emb = get_baseline_embedding(
        eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]  # (1, d)

    log_odds_sum, comps_sum, suffs_sum, count, total_time = 0.0, 0.0, 0.0, 0, 0.0

    for text, _label in tqdm(data, desc=f"  baseline={BASELINE_LABEL[baseline]}", leave=False):
        res = fcg_gradient_classification(
            sentence=text, a=a, b=b, steps=steps,
            model_name=model_name,
            show_special_tokens=False,
            baseline=baseline,
            device=device,
        )

        attr = res["attr_full"]

        log_odd, _ = calculate_log_odds(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, topk=20,
        )
        comp = calculate_soft_comprehensiveness(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, n_samples=n_samples,
        )
        suff = calculate_soft_sufficiency(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, n_samples=n_samples,
        )

        log_odds_sum += log_odd
        comps_sum    += comp
        suffs_sum    += suff
        total_time   += res["time"]
        count        += 1

    return {
        "log_odds":  log_odds_sum / count,
        "soft_comp": comps_sum    / count,
        "soft_suff": suffs_sum    / count,
        "avg_time":  total_time   / count,
        "count":     count,
    }


def print_table(results: dict, dataset_name: str, model_name: str):
    """Print a formatted comparison table for a single (model, dataset)."""
    print(f"\n{'='*90}")
    print(f"  Model: {model_name}  |  Dataset: {dataset_name}")
    print(f"{'='*90}")
    header = f"  {'Baseline':<14s}  {'Log-Odds':>10s}  {'Soft-Comp':>10s}  {'Soft-Suff':>10s}  {'Time(s)':>10s}"
    print(header)
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for baseline in BASELINES:
        r = results[baseline]
        print(
            f"  {BASELINE_LABEL[baseline]:<14s}  "
            f"{r['log_odds']:>10.4f}  "
            f"{r['soft_comp']:>10.4f}  "
            f"{r['soft_suff']:>10.4f}  "
            f"{r['avg_time']:>10.4f}"
        )
    print(f"  {'='*90}")


def print_summary_table(all_results: dict):
    """Print a global summary across all models and datasets."""
    print(f"\n{'='*110}")
    print(f"  GLOBAL SUMMARY — Averaged across datasets per model")
    print(f"{'='*110}")
    header = f"  {'Model':<14s}  {'Baseline':<14s}  {'Log-Odds':>10s}  {'Soft-Comp':>10s}  {'Soft-Suff':>10s}"
    print(header)
    print(f"  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")

    for model_name in ["distilbert", "bert", "roberta"]:
        for baseline in BASELINES:
            # Average across datasets
            lo = np.mean([all_results[model_name][ds][baseline]["log_odds"] for ds in ["sst2", "imdb", "rotten"]])
            sc = np.mean([all_results[model_name][ds][baseline]["soft_comp"] for ds in ["sst2", "imdb", "rotten"]])
            ss = np.mean([all_results[model_name][ds][baseline]["soft_suff"] for ds in ["sst2", "imdb", "rotten"]])
            print(
                f"  {model_name:<14s}  {BASELINE_LABEL[baseline]:<14s}  "
                f"{lo:>10.4f}  {sc:>10.4f}  {ss:>10.4f}"
            )
        if model_name != "roberta":
            print(f"  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'='*110}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: effect of reference embedding token on PG"
    )
    parser.add_argument("--models", type=str, nargs="+",
                        default=["distilbert", "bert", "roberta"],
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["sst2", "imdb", "rotten"],
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--baselines", type=str, nargs="+",
                        default=BASELINES,
                        choices=BASELINES,
                        help="Reference embedding types to ablate over")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline for faithfulness metric computation (fixed)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Stochastic samples for soft metrics")
    args = parser.parse_args()

    a, b          = 0, 1
    steps         = args.steps
    eval_baseline = args.eval_baseline
    n_samples     = args.n_samples
    # device        = "cuda" if torch.cuda.is_available() else "cpu"
    device        = "cpu"  # force CPU for reproducibility
    print(f"Device        : {device}")
    print(f"Models        : {args.models}")
    print(f"Datasets      : {args.datasets}")
    print(f"FCG baselines  : {[BASELINE_LABEL[b] for b in args.baselines]}")
    print(f"Eval baseline : {BASELINE_LABEL[eval_baseline]} (fixed)")
    print(f"Range         : [{a}, {b}]  steps={steps}")
    print(f"Soft samples  : {n_samples}")

    # all_results[model][dataset][baseline] = metrics_dict
    all_results = {}

    for model_name in args.models:
        all_results[model_name] = {}
        for dataset_name in args.datasets:
            print(f"\n{'#'*90}")
            print(f"#  MODEL={model_name}  DATASET={dataset_name}")
            print(f"{'#'*90}")

            data = load_dataset_split(dataset_name)
            hf_model_name = MODEL_CONFIGS[model_name][dataset_name]
            all_results[model_name][dataset_name] = {}

            for baseline in args.baselines:
                print(f"\n--- FCG baseline: {BASELINE_LABEL[baseline]} ({baseline}) ---")
                result = evaluate_baseline(
                    model_name=hf_model_name,
                    data=data,
                    baseline=baseline,
                    eval_baseline=eval_baseline,
                    steps=steps,
                    n_samples=n_samples,
                    device=device,
                    a=a, b=b,
                )
                all_results[model_name][dataset_name][baseline] = result
                print(
                    f"  [{BASELINE_LABEL[baseline]:>10s}]  "
                    f"Log-odds: {result['log_odds']:.4f}  "
                    f"Soft-Comp: {result['soft_comp']:.4f}  "
                    f"Soft-Suff: {result['soft_suff']:.4f}  "
                    f"Time: {result['avg_time']:.4f}s  "
                    f"(n={result['count']})"
                )

            # Print per (model, dataset) comparison table
            print_table(all_results[model_name][dataset_name], dataset_name, model_name)

    # Print global summary
    if len(args.models) > 1 and len(args.datasets) > 1:
        print_summary_table(all_results)

    print("\nDone.")
