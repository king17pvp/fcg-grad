"""
run_eval_fi_gpt_new.py

Benchmark FI attribution variants on decoder-only GPT models using the three
evaluation settings from Zhao & Shan (ReAGent, AAAI 2024):

    arxiv.org/abs/2402.00794

Supported experiments (--experiment):
--------------------------------------
  tellmewhy   Sequence-level faithfulness on TellMeWhy why-question narratives.
              Soft-NS/NC computed every 5 generated tokens, averaged.
              Expected file format (one sample per line):
                  <narrative + why-question>[\t<gold_answer>]

  longra      Token-level faithfulness on Long-Range Agreement templates.
              Soft-NS/NC computed on the single target word position only.
              Only keeps samples where the model predicts the same target
              word with and without the distractor sentence (in parentheses).
              Expected file format (one sample per line):
                  <full_prompt_with_distractor>\t<target_word>

  wikibio     Sequence-level faithfulness on Wikipedia biography continuations.
              First two sentences are used as the prompt; the model continues
              the biography. Soft-NS/NC computed every 5 tokens, averaged.
              Expected file format (one sample per line):
                  <first_two_biography_sentences>

Supported FI methods (--method):
----------------------------------
  fi            Standard FI: E[(∇f)²/f]               (default)
  fi_cov        Covariance FI: E[(Σ∇f)·∇f/f]
  smooth_grad   SmoothGrad: E[∇f]
  smooth_grad_sq SmoothGradSQ: E[(∇f)²]
  all           Run all four methods in sequence

Supported models (--model_name):
----------------------------------
  GPT-2 family : gpt2 | gpt2-medium | gpt2-large | gpt2-xl
  OPT family   : facebook/opt-350m | facebook/opt-1.3b | facebook/opt-6.7b
  GPT-J        : EleutherAI/gpt-j-6b
  Any other AutoModelForCausalLM-compatible model on HuggingFace.

Metrics (following ReAGent, Zhao & Shan 2024):
  Soft-NC  -- Soft Normalised Comprehensiveness  (Hellinger-based, Eq.15)
  Soft-NS  -- Soft Normalised Sufficiency        (Hellinger-based, Eq.14)
  Log-odds -- token-level log-probability drop after masking top-k Q tokens

Usage examples:
    python run_eval_fi_gpt_new.py --experiment tellmewhy --model_name gpt2
    python run_eval_fi_gpt_new.py --experiment longra    --model_name gpt2 --method fi
    python run_eval_fi_gpt_new.py --experiment wikibio   --model_name gpt2-medium --method all
    python run_eval_fi_gpt_new.py --experiment tellmewhy --model_name facebook/opt-1.3b
    python run_eval_fi_gpt_new.py --experiment longra    --model_name EleutherAI/gpt-j-6b --num_samples 36
"""

import re
import random
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fi_gpt2 import fi_gradient_gpt2, _build_base_embed
from xai_metrics_gpt2 import calculate_all_metrics_gpt2

#  reproducibility 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

ALL_METHODS = ["fi", "fi_cov", "smooth_grad", "smooth_grad_sq"]


# 
# Experiment metadata
# 

EXPERIMENT_META = {
    "tellmewhy": {
        "description": "Sequence-level faithfulness — TellMeWhy why-question narratives",
        "default_path": "datasets2/tellmewhy2_new.txt",
        "eval_mode":    "sequence",   # Soft-NS/NC over every 5 answer tokens
        "stride":       5,
    },
    "longra": {
        "description": "Token-level faithfulness — Long-Range Agreement word-pair templates",
        "default_path": "datasets2/longra.txt",
        "eval_mode":    "token",      # Soft-NS/NC on the single target word position
        "stride":       1,
    },
    "wikibio": {
        "description": "Sequence-level faithfulness — Wikipedia biography continuation",
        "default_path": "datasets2/wikibio.txt",
        "eval_mode":    "sequence",
        "stride":       5,
    },
}


# 
# Generic model / tokenizer loading (supports GPT-2, OPT, GPT-J, ...)
# 

_MODEL_CACHE: dict = {}

def load_model_tokenizer(model_name: str, device: str):
    """
    Load any causal-LM from HuggingFace, with a process-level cache.
    Falls back to AutoModelForCausalLM / AutoTokenizer so that OPT and
    GPT-J models work alongside GPT-2 variants.
    """
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    print(f"  Loading tokenizer : {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Loading model     : {model_name}")
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    mdl.eval().to(device)

    _MODEL_CACHE[key] = (mdl, tok)
    return mdl, tok


def get_embed_layer(model):
    """
    Return the word-token embedding layer for GPT-2, OPT, or GPT-J models.
    Tries common attribute paths.
    """
    for attr in ("transformer.wte", "model.decoder.embed_tokens", "model.embed_tokens"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"Cannot locate embedding layer in {type(model).__name__}. "
        "Add the correct attribute path to get_embed_layer()."
    )


# 
# Dataset loaders
# 

def load_tellmewhy(path: str, num_samples: int, use_gold: bool) -> list[dict]:
    """
    Load TellMeWhy samples from a plain-text file.

    File format — one sample per line, optionally tab-separated:
        <narrative + why-question>[\t<gold_answer>]
    """
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts     = line.split("\t")
            full_text = parts[0].strip()
            gold_ans  = parts[1].strip() if (use_gold and len(parts) > 1) else None
            if full_text:
                samples.append({"question": full_text,
                                 "gold_answer": gold_ans,
                                 "target_word": None})

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    print(f"Loaded {len(samples)} TellMeWhy samples from {path}")
    return samples


def load_longra(path: str, num_samples: int, model, tokenizer, device: str) -> list[dict]:
    """
    Load Long-Range Agreement samples and keep only those for which the model
    produces the same prediction with and without the distractor sentence.

    File format — one sample per line, tab-separated:
        <full_prompt_with_distractor_in_parentheses>\t<target_word>
    """
    raw_samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            prompt      = parts[0].strip()
            target_word = parts[1].strip()
            if prompt and target_word:
                raw_samples.append({"prompt": prompt, "target_word": target_word})

    _DIST_RE = re.compile(r"\s*\([^)]*\)\s*", re.DOTALL)

    def _greedy_next_token(text: str) -> str:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=1, do_sample=False)
        new_id = out[0, ids.shape[1]:]
        return tokenizer.decode(new_id, skip_special_tokens=True).strip()

    samples = []
    for s in raw_samples:
        prompt_no_dist = _DIST_RE.sub(" ", s["prompt"]).strip()
        pred_with      = _greedy_next_token(s["prompt"])
        pred_without   = _greedy_next_token(prompt_no_dist)
        target_lc      = s["target_word"].lower()
        if target_lc in pred_with.lower() and target_lc in pred_without.lower():
            samples.append({
                "question":    s["prompt"],
                "gold_answer": None,
                "target_word": s["target_word"],
            })

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    print(f"Loaded {len(samples)} valid LongRA samples from {path} "
          f"(filtered from {len(raw_samples)} raw entries)")
    return samples


def load_wikibio(path: str, num_samples: int) -> list[dict]:
    """
    Load Wikipedia biography prompts from a plain-text file.

    File format — one entry per line:
        <first two sentences of the biography>
    """
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            samples.append({"question": line,
                             "gold_answer": None,
                             "target_word": None})

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    print(f"Loaded {len(samples)} WikiBio samples from {path}")
    return samples


# 
# Single-sample pipeline
# 

def run_single_example(
    sample: dict,
    model_name: str,
    device: str,
    method: str,
    n: int,
    var_spread: float,
    topk: int,
    max_new_tokens: int,
    mc_samples: int,
    baseline: str,
    eval_base_embed: torch.Tensor,   # (1, 1, D)
    stride: int,
    eval_mode: str,
) -> dict:
    """
    Run FI attribution and compute faithfulness metrics for one sample.

    Parameters
    ----------
    eval_mode : 'sequence' → Soft-NS/NC over every `stride` answer tokens
                'token'    → Soft-NS/NC on the first answer token only
                             (used for LongRA target-word evaluation)
    """
    res = fi_gradient_gpt2(
        question       = sample["question"],
        model_name     = model_name,
        device         = device,
        n              = n,
        var_spread     = var_spread,
        max_new_tokens = max_new_tokens,
        gold_answer    = sample["gold_answer"],
        method         = method,
        baseline       = baseline,
    )

    # For LongRA token-level: evaluate ONLY at the first generated token position.
    answer_positions = res["answer_positions"]
    if eval_mode == "token":
        answer_positions = answer_positions[:1]
        effective_stride = 1
    else:
        effective_stride = stride

    # Expand eval_base_embed from (1, 1, D) to (1, T, D) for this sample
    T         = res["input_embed"].shape[1]
    eval_base = eval_base_embed.expand(1, T, -1)   # broadcast, no copy

    metrics = calculate_all_metrics_gpt2(
        model            = res["model"],
        input_embed      = res["input_embed"],
        base_embed       = eval_base,
        attributions     = res["attributions"],
        answer_ids       = res["answer_ids"],
        answer_positions = answer_positions,
        topk             = topk,
        n_samples        = mc_samples,
        device           = device,
        eval_base_embed  = eval_base,
        stride           = effective_stride,
    )

    return {
        "tokens":           res["tokens"],
        "q_len":            res["q_len"],
        "attributions":     res["attributions"],
        "predicted_answer": res["predicted_answer"],
        "time":             res["time"],
        "soft_nc":          metrics["soft_nc"].item(),
        "soft_ns":          metrics["soft_ns"].item(),
        "log_odds":         metrics["log_odds"].item(),
    }


# 
# Per-method benchmark loop
# 

def run_benchmark_single(
    args, method: str, samples: list,
    device: str, eval_base_embed: torch.Tensor,
    stride: int, eval_mode: str,
) -> dict:
    total_soft_nc  = 0.0
    total_soft_ns  = 0.0
    total_log_odds = 0.0
    total_time     = 0.0
    count = errors = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"[FI-{method}/{args.experiment}]")):
        try:
            res = run_single_example(
                sample          = sample,
                model_name      = args.model_name,
                device          = device,
                method          = method,
                n               = args.n,
                var_spread      = args.var_spread,
                topk            = args.topk,
                max_new_tokens  = args.max_new_tokens,
                mc_samples      = args.mc_samples,
                baseline        = args.baseline,
                eval_base_embed = eval_base_embed,
                stride          = stride,
                eval_mode       = eval_mode,
            )

            total_soft_nc  += res["soft_nc"]
            total_soft_ns  += res["soft_ns"]
            total_log_odds += res["log_odds"]
            total_time     += res["time"]
            count          += 1

            if args.verbose and count <= 3:
                _print_sample(sample["question"], res)

            if count % args.print_step == 0:
                _print_running(method, count, len(samples),
                               total_soft_nc, total_soft_ns,
                               total_log_odds, total_time)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"\n[Error sample {idx}]: {str(exc)[:120]}")
                traceback.print_exc()

    return {
        "method":   method,
        "count":    count,
        "errors":   errors,
        "soft_nc":  total_soft_nc  / max(count, 1),
        "soft_ns":  total_soft_ns  / max(count, 1),
        "log_odds": total_log_odds / max(count, 1),
        "avg_time": total_time     / max(count, 1),
    }


# 
# Benchmark runner
# 

def run_benchmark(args) -> None:
    experiment = args.experiment
    meta       = EXPERIMENT_META[experiment]
    methods    = ALL_METHODS if args.method == "all" else [args.method]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(torch.device(device))

    print("=" * 60)
    print(f"Method(s)     : FI  {methods}")
    print(f"Experiment    : {experiment}  —  {meta['description']}")
    print(f"Eval mode     : {meta['eval_mode']}  (stride={meta['stride']})")
    print(f"Device        : {device}")
    print(f"Model         : {args.model_name}")
    print(f"Dataset path  : {args.data_path or meta['default_path']}")
    print(f"Samples       : {args.num_samples}")
    print(f"MC draws n    : {args.n}")
    print(f"var_spread    : {args.var_spread}")
    print(f"Top-k %       : {args.topk}")
    print(f"Baseline      : {args.baseline}")
    print(f"Eval baseline : {args.eval_baseline}")
    print(f"MC samples    : {args.mc_samples}")
    print("=" * 60)

    #  Model loading 
    print("\nLoading model ...")
    model, tokenizer = load_model_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    # Register in fi_gpt2's cache to avoid reloading on every call
    from fi_gpt2 import _CACHE as _fi_cache
    _fi_cache[(args.model_name, device)] = (model, tokenizer)

    #  Build eval_base_embed once  (1, 1, D), broadcast to (1, T, D) per sample
    embed_layer = get_embed_layer(model)
    with torch.no_grad():
        dummy_embed = embed_layer(
            torch.tensor([[tokenizer.eos_token_id]], device=device)
        ).detach().cpu()   # (1, 1, D)

    eval_base_embed = _build_base_embed(
        embed_layer, dummy_embed,
        args.eval_baseline, tokenizer.eos_token_id,
        device="cpu",
    )   # (1, 1, D)

    #  Dataset loading 
    data_path = args.data_path or meta["default_path"]

    if experiment == "tellmewhy":
        samples = load_tellmewhy(
            data_path, num_samples=args.num_samples, use_gold=args.use_gold
        )
    elif experiment == "longra":
        samples = load_longra(
            data_path, num_samples=args.num_samples,
            model=model, tokenizer=tokenizer, device=device,
        )
    elif experiment == "wikibio":
        samples = load_wikibio(data_path, num_samples=args.num_samples)
    else:
        raise ValueError(f"Unknown experiment: {experiment!r}")

    if not samples:
        print("No samples found — check --data_path.")
        return

    #  Run each method 
    all_results = []
    for method in methods:
        result = run_benchmark_single(
            args, method, samples, device, eval_base_embed,
            stride=meta["stride"], eval_mode=meta["eval_mode"],
        )
        all_results.append(result)

    #  Final report 
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print(f"  Experiment : {experiment}  ({meta['eval_mode']}-level, "
          f"stride={meta['stride']})")
    print(f"  Model      : {args.model_name}")
    print("=" * 70)
    print(f"{'Method':<20} {'Soft-NC':>10} {'Soft-NS':>10} {'Log-odds':>10} {'Time(s)':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:<20} {r['soft_nc']:>10.6f} {r['soft_ns']:>10.6f}"
              f" {r['log_odds']:>10.6f} {r['avg_time']:>10.4f}")
        print(f"  Successful: {r['count']} / {len(samples)}    Errors: {r['errors']}")
    print("=" * 70)


# 
# Printing helpers
# 

def _print_running(method, count, total, snc, sns, lo, t):
    print(f"\n[{method}] [{count}/{total}]  "
          f"Soft-NC={snc/count:.4f}  Soft-NS={sns/count:.4f}  "
          f"Log-odds={lo/count:.4f}  AvgTime={t/count:.4f}s")


def _print_sample(question: str, res: dict):
    tokens = res["tokens"]
    scores = res["attributions"].tolist()
    q_len  = res["q_len"]

    print(f"\n{'' * 60}")
    print(f"Q : {question[:120]}")
    print(f"A : {res['predicted_answer']}")

    q_scores = sorted(zip(tokens[:q_len], scores[:q_len]),
                      key=lambda x: x[1], reverse=True)
    print("Top-5 Q tokens by FI attribution:")
    for tok, sc in q_scores[:5]:
        print(f"    {tok!r:20s}  {sc:.6f}")

    print(f"Soft-NC={res['soft_nc']:.4f}  "
          f"Soft-NS={res['soft_ns']:.4f}  "
          f"Log-odds={res['log_odds']:.4f}  "
          f"Time={res['time']:.2f}s")


# 
# CLI
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FI Attribution on GPT-family models "
            "following the three experiments in Zhao & Shan (ReAGent, AAAI 2024)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments
-----------
  tellmewhy   Why-question narratives (sequence-level Soft-NS/NC, stride=5)
  longra      Word-pair templates     (token-level   Soft-NS/NC, target word only)
  wikibio     Biography continuation  (sequence-level Soft-NS/NC, stride=5)

FI methods
----------
  fi             Standard FI: E[(∇f)²/f]
  fi_cov         Covariance FI: E[(Σ∇f)·∇f/f]
  smooth_grad    SmoothGrad: E[∇f]
  smooth_grad_sq SmoothGradSQ: E[(∇f)²]
  all            Run all four methods in sequence

Dataset file formats
--------------------
  tellmewhy : <narrative + why-question>[TAB<gold_answer>]  — one sample per line
  longra    : <full_prompt_with_(distractor)>[TAB<target_word>]  — one sample per line
  wikibio   : <first two biography sentences>  — one entry per line
""",
    )

    #  Experiment selection 
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(EXPERIMENT_META.keys()),
        help="Which evaluation experiment to run: tellmewhy | longra | wikibio",
    )

    #  Data / model 
    parser.add_argument(
        "--data_path", type=str, default=None,
        help=(
            "Path to the dataset file. If omitted, each experiment uses "
            "its built-in default path (see EXPERIMENT_META)."
        ),
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help=(
            "HuggingFace model identifier. "
            "GPT-2 family : gpt2 | gpt2-medium | gpt2-large | gpt2-xl. "
            "OPT family   : facebook/opt-350m | facebook/opt-1.3b | facebook/opt-6.7b. "
            "GPT-J        : EleutherAI/gpt-j-6b. "
            "(default: gpt2)"
        ),
    )

    #  FI method 
    parser.add_argument(
        "--method", type=str, default="fi",
        choices=ALL_METHODS + ["all"],
        help="FI attribution variant to run (default: fi)",
    )

    #  Sampling & generation 
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Max samples to evaluate (default: 200; paper uses ~36 for LongRA)",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Monte-Carlo perturbation draws for FI estimate (default: 20)",
    )
    parser.add_argument(
        "--var_spread", type=float, default=0.15,
        help="Multiplier on per-dim variance for noise σ² (default: 0.15)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=30,
        help="Max tokens to generate per answer (default: 30)",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Percentage of top Q-tokens to mask for log-odds (default: 20)",
    )
    parser.add_argument(
        "--mc_samples", type=int, default=10,
        help="Monte-Carlo draws for soft Bernoulli perturbation metrics (default: 10)",
    )

    #  Baseline options 
    parser.add_argument(
        "--baseline", type=str, default="zero",
        choices=["zero", "pad", "mean"],
        help="Baseline for FI method base_embed reference (default: zero)",
    )
    parser.add_argument(
        "--eval-baseline", type=str, default="zero",
        dest="eval_baseline",
        choices=["zero", "pad", "mean"],
        help="Baseline embedding used in ΔP_0 normalisation anchor for Soft-NC/NS (default: zero)",
    )

    #  Misc 
    parser.add_argument(
        "--use_gold", action="store_true",
        help="(TellMeWhy only) Use tab-separated gold answers from the file if available",
    )
    parser.add_argument(
        "--print_step", type=int, default=50,
        help="Print running averages every N samples (default: 50)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print attribution details for the first 3 samples",
    )

    args = parser.parse_args()
    run_benchmark(args)
