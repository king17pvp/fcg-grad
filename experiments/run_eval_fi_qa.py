"""
run_eval_fi_qa.py

Functional Information (FI) attribution for Question Answering models.

FI variants supported (same as fi_classification / fi_gpt2.py):
  fi            — E[(∇f)² / f]          (standard, default)
  smooth_grad   — E[∇f]
  smooth_grad_sq — E[(∇f)²]
  fi_cov        — E[(Σ∇f)·∇f / f]

For QA, attributions are computed separately for start and end logits:
  f_start(z) = softmax(start_logits)[pred_start_idx]   (scalar, always > 0)
  f_end(z)   = softmax(end_logits)[pred_end_idx]       (scalar, always > 0)

Evaluation metrics: log-odds, soft-comprehensiveness, soft-sufficiency (QA variants).

Mirrors run_eval_pg_qa.py structure — only the attribution backend changes.

Usage:
  python run_eval_fi_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 1000
  python run_eval_fi_qa.py --method fi_cov --n 30 --eval-baseline zero
"""

import time
import random
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from fcg_gradients import get_baseline_embedding
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

#  Model cache 
_cache = {}


def _load_qa(model_name: str, device: str):
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        model.eval()
        _cache[model_name] = {"model": model, "tokenizer": tokenizer}
    entry = _cache[model_name]
    return entry["model"], entry["tokenizer"]


#  Covariance helpers (mirrors fi_classification) 

def _estimate_covariance(embed: torch.Tensor, regularise: float = 1e-4) -> torch.Tensor:
    """Estimate a shared (D, D) covariance from embed [1, L, D]."""
    flat = embed.squeeze(0)           # [L, D]
    B, D = flat.shape
    if B < 2:
        var = flat.var(dim=0).clamp(min=regularise)
        return torch.diag(var)
    xc  = flat - flat.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / (B - 1)
    cov += regularise * torch.eye(D, device=flat.device, dtype=flat.dtype)
    return cov


def _cholesky_safe(cov: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        d = torch.abs(torch.diag(cov)).clamp(min=1e-6)
        try:
            return torch.linalg.cholesky(cov + d.min() * torch.eye(cov.shape[0], device=cov.device))
        except RuntimeError:
            return torch.linalg.cholesky(cov + d.mean() * torch.eye(cov.shape[0], device=cov.device))


#  Core FI QA attribution 

def fi_qa(
    question: str,
    context: str,
    model_name: str,
    device: str = "cpu",
    n: int = 20,
    var_spread: float = 0.15,
    method: str = "fi",
    show_special_tokens: bool = False,
) -> dict:
    """
    Functional Information attribution for a BERT-based QA model.

    f_start(z) = softmax(start_logits)[pred_start]  — scalar, always > 0
    f_end(z)   = softmax(end_logits)[pred_end]      — scalar, always > 0

    Args:
        question   : Question string.
        context    : Passage/context string.
        model_name : HuggingFace QA model name.
        device     : Torch device.
        n          : Monte-Carlo perturbation draws.
        var_spread : Noise scale multiplier on per-dim embedding variance.
        method     : 'fi' | 'smooth_grad' | 'smooth_grad_sq' | 'fi_cov'
        show_special_tokens: Whether to include [CLS]/[SEP] in output.

    Returns dict with keys:
        tokens, attributions_start, attributions_end, predicted_answer,
        start_idx, end_idx, start_logit, end_logit, start_prob, end_prob,
        model, input_embed, attention_mask, special_tokens_mask,
        token_type_ids, base_token_emb, time
    """
    t0 = time.perf_counter()

    model, tokenizer = _load_qa(model_name, device)

    #  Tokenize (question + context) 
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
    )
    input_ids           = enc["input_ids"].to(device)                # (1, L)
    attention_mask      = enc["attention_mask"].to(device)           # (1, L)
    token_type_ids      = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    special_tokens_mask = enc["special_tokens_mask"][0].to(device).bool()  # (L,) bool
    seq_len             = input_ids.shape[1]

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)   # (1, L, D)

    L, D = X.shape[1], X.shape[2]

    #  Predicted span 
    with torch.no_grad():
        out0 = model(
            inputs_embeds=X, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    start_idx = int(out0.start_logits.argmax(dim=-1).item())
    end_idx   = int(out0.end_logits.argmax(dim=-1).item())
    if end_idx < start_idx:
        end_idx = start_idx

    prob_start_orig = F.softmax(out0.start_logits, dim=-1)[0, start_idx]
    prob_end_orig   = F.softmax(out0.end_logits,   dim=-1)[0, end_idx]
    start_logit_val = out0.start_logits[0, start_idx].item()
    end_logit_val   = out0.end_logits[0, end_idx].item()

    #  Decode predicted answer 
    predicted_answer = tokenizer.decode(
        input_ids[0, start_idx:end_idx + 1], skip_special_tokens=True,
    )

    #  Per-dim noise variance 
    sigma2_vec = X.squeeze(0).var(dim=0) * var_spread   # (D,)

    # Covariance for fi_cov
    if method == "fi_cov":
        Sigma = _estimate_covariance(X) * var_spread    # (D, D)
        Sigma = Sigma.to(device)

    #  Monte-Carlo accumulation for start and end 
    accumulated_start = torch.zeros(1, L, D, device=device, dtype=X.dtype)
    accumulated_end   = torch.zeros(1, L, D, device=device, dtype=X.dtype)

    for _ in range(n):
        noise = torch.randn_like(X) * sigma2_vec.sqrt()
        z     = (X + noise).requires_grad_(True)

        out_z = model(
            inputs_embeds=z, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        #  Start position gradient 
        f_start = F.softmax(out_z.start_logits, dim=-1)[0, start_idx].clamp(min=1e-8)
        model.zero_grad()
        f_start.backward(retain_graph=True)
        grad_start = z.grad.detach().clone()   # (1, L, D)
        z.grad = None

        #  End position gradient 
        f_end = F.softmax(out_z.end_logits, dim=-1)[0, end_idx].clamp(min=1e-8)
        model.zero_grad()
        f_end.backward()
        grad_end = z.grad.detach().clone()     # (1, L, D)

        if method == "fi":
            accumulated_start += (grad_start ** 2) / f_start.detach()
            accumulated_end   += (grad_end ** 2)   / f_end.detach()

        elif method == "smooth_grad":
            accumulated_start += grad_start
            accumulated_end   += grad_end

        elif method == "smooth_grad_sq":
            accumulated_start += (grad_start ** 2)
            accumulated_end   += (grad_end ** 2)

        elif method == "fi_cov":
            for accum, grad, f_val in [
                (accumulated_start, grad_start, f_start),
                (accumulated_end,   grad_end,   f_end),
            ]:
                g       = grad.squeeze(0)                                    # (L, D)
                sigma_g = (Sigma @ g.unsqueeze(-1)).squeeze(-1)              # (L, D)
                fi_cov  = (sigma_g * g) / f_val.detach()                    # (L, D)
                accum.add_(fi_cov.unsqueeze(0))

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose: fi | fi_cov | smooth_grad | smooth_grad_sq"
            )

    #  L2-norm across embedding dims → token attribution 
    mean_attr_start = accumulated_start / n                            # (1, L, D)
    mean_attr_end   = accumulated_end / n                              # (1, L, D)
    attr_start_full = mean_attr_start.norm(dim=-1).squeeze(0).detach() # (L,)
    attr_end_full   = mean_attr_end.norm(dim=-1).squeeze(0).detach()   # (L,)

    #  Token output 
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if show_special_tokens:
        tokens     = tokens_raw
        attr_s_out = attr_start_full
        attr_e_out = attr_end_full
    else:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        tokens     = [tokens_raw[i] for i in keep_idx]
        attr_s_out = attr_start_full[keep_idx]
        attr_e_out = attr_end_full[keep_idx]

    #  base_token_emb for metrics (externally overridden in eval script) 
    with torch.no_grad():
        base_token_emb = embed(
            torch.tensor([[tokenizer.mask_token_id or tokenizer.pad_token_id]], device=device)
        )[:, 0, :]   # (1, D)

    return {
        "tokens":              tokens,
        "attributions_start":  attr_s_out.detach().cpu(),
        "attributions_end":    attr_e_out.detach().cpu(),
        "predicted_answer":    predicted_answer,
        "start_idx":           start_idx,
        "end_idx":             end_idx,
        "start_logit":         start_logit_val,
        "end_logit":           end_logit_val,
        "start_prob":          prob_start_orig,     # Tensor — for xai_metrics
        "end_prob":            prob_end_orig,       # Tensor — for xai_metrics
        # Full-length tensors for metrics
        "attr_start_full":     attr_start_full,
        "attr_end_full":       attr_end_full,
        # Tensors required by xai_metrics QA functions
        "model":               model,
        "input_embed":         X,
        "attention_mask":      attention_mask,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids":      token_type_ids,
        "base_token_emb":      base_token_emb,
        "time":                time.perf_counter() - t0,
    }


#  Single-example runner (mirrors run_eval_pg_qa.run_single_example) 

def run_single_example(
    question: str,
    context: str,
    model_name: str,
    device: str,
    eval_base_token_emb: torch.Tensor,
    n: int = 20,
    var_spread: float = 0.15,
    method: str = "fi",
    topk: int = 20,
    n_samples: int = 10,
) -> dict:
    """
    Run FI-QA attribution on one example and compute faithfulness metrics.
    """
    res = fi_qa(
        question=question,
        context=context,
        model_name=model_name,
        device=device,
        n=n,
        var_spread=var_spread,
        method=method,
        show_special_tokens=True,
    )

    model               = res["model"]
    input_embed         = res["input_embed"]
    attention_mask      = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids      = res["token_type_ids"]
    attr_start          = res["attr_start_full"]
    attr_end            = res["attr_end_full"]
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
        "tokens":             res["tokens"],
        "attributions_start": res["attributions_start"],
        "attributions_end":   res["attributions_end"],
        "predicted_answer":   res["predicted_answer"],
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "start_logit":        res["start_logit"],
        "end_logit":          res["end_logit"],
        "time":               res["time"],
        "log_odd_start":      log_odd_start,
        "soft_comp_start":    soft_comp_start,
        "soft_suff_start":    soft_suff_start,
        "log_odd_end":        log_odd_end,
        "soft_comp_end":      soft_comp_end,
        "soft_suff_end":      soft_suff_end,
    }


#  Benchmark loop 

def run_benchmark(args):
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model         : {args.model_name}")
    print(f"Dataset       : {args.dataset}")
    print(f"FI method     : {args.method}")
    print(f"n draws       : {args.n}")
    print(f"var_spread    : {args.var_spread}")
    print(f"Eval baseline : {args.eval_baseline}")
    print(f"Soft samples  : {args.n_samples}")
    print(f"Top-k         : {args.topk}%")

    #  Build eval_base_token_emb once 
    model, tokenizer = _load_qa(args.model_name, device)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, D)
    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, D)

    #  Smoke test 
    demo_q = "Who invented the telephone?"
    demo_c = (
        "Alexander Graham Bell is often credited with inventing the telephone "
        "in 1876, though Elisha Gray filed a patent caveat on the same day."
    )
    print("\n--- FI QA demo attribution ---")
    demo = fi_qa(
        demo_q, demo_c, model_name=args.model_name, device=device,
        n=args.n, var_spread=args.var_spread, method=args.method,
        show_special_tokens=False,
    )
    print(f"Predicted answer: {demo['predicted_answer']!r}")
    print(f"Start idx={demo['start_idx']}  End idx={demo['end_idx']}")
    print("\nToken attributions (start | end):")
    for tok, s, e in zip(
        demo["tokens"],
        demo["attributions_start"].tolist(),
        demo["attributions_end"].tolist(),
    ):
        print(f"  {tok:>25s} : start={s:+.5f}  end={e:+.5f}")

    #  Dataset 
    print(f"\nLoading dataset: {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="validation")
    data = list(zip(
        dataset["question"], dataset["context"],
        dataset["answers"],  dataset["id"],
    ))

    # Same filter as PG QA
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
                device=device,
                eval_base_token_emb=eval_base_token_emb,
                n=args.n,
                var_spread=args.var_spread,
                method=args.method,
                topk=args.topk,
                n_samples=args.n_samples,
            )

            total_log_odds_start  += res["log_odd_start"]
            total_log_odds_end    += res["log_odd_end"]
            total_soft_comp_start += res["soft_comp_start"]
            total_soft_comp_end   += res["soft_comp_end"]
            total_soft_suff_start += res["soft_suff_start"]
            total_soft_suff_end   += res["soft_suff_end"]
            total_time            += res["time"]
            count                 += 1

            if count % args.print_step == 0:
                print(
                    f"\n[{count}/{len(sampled_data)}] Running averages:")
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
        print(f"FI-QA ({args.method}, n={args.n})  |  {args.model_name}")
        print(f"  Log-odds (start)              : {total_log_odds_start/count:.6f}")
        print(f"  Soft-Comprehensiveness (start): {total_soft_comp_start/count:.6f}")
        print(f"  Soft-Sufficiency (start)      : {total_soft_suff_start/count:.6f}")
        print(f"  Log-odds (end)                : {total_log_odds_end/count:.6f}")
        print(f"  Soft-Comprehensiveness (end)  : {total_soft_comp_end/count:.6f}")
        print(f"  Soft-Sufficiency (end)        : {total_soft_suff_end/count:.6f}")
        print(f"  Avg time/sample               : {total_time/count:.4f}s")
        print(f"  Evaluated                     : {count}  |  Errors: {errors}")
        print(f"{''*52}")


#  Entry point 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FI Attribution for Question Answering"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="deepset/bert-base-cased-squad2",
        help="HuggingFace QA model name",
    )
    parser.add_argument(
        "--dataset", type=str, default="squad",
        help="Dataset name from HuggingFace datasets",
    )
    parser.add_argument(
        "--method", type=str, default="fi",
        choices=["fi", "smooth_grad", "smooth_grad_sq", "fi_cov"],
        help="FI variant to evaluate",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Monte-Carlo perturbation draws",
    )
    parser.add_argument(
        "--var_spread", type=float, default=0.15,
        help="Noise scale multiplier on per-dim embedding variance",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--topk", type=int, default=50,
        help="Percentage of top tokens for metrics calculation",
    )
    parser.add_argument(
        "--print_step", type=int, default=100,
        help="Print running averages every N samples",
    )
    parser.add_argument(
        "--eval-baseline", type=str, default="mask",
        choices=["mask", "pad", "zero", "mean", "random"],
        help="Baseline embedding used to replace tokens in faithfulness metrics",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Stochastic samples for soft metrics",
    )
    args = parser.parse_args()
    run_benchmark(args)
