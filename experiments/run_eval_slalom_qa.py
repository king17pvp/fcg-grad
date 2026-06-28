"""
run_slalom_eval_qa.py

Benchmark SLALOM explanations on Question Answering datasets (SQuAD / SQuADv2)
with the same metrics interface as run_eval_pg_qa.py
(log-odds, comprehensiveness, sufficiency — separate for start and end positions).

Usage:
    python run_slalom_eval_qa.py --model_name deepset/bert-base-cased-squad2 --dataset squad
    python run_slalom_eval_qa.py --model_name deepset/bert-base-cased-squad2 --dataset squad_v2 --num_samples 500
    python run_slalom_eval_qa.py --model_name deepset/bert-base-cased-squad2 --dataset squad --attr_mode value
"""

import time
import random
import argparse
import traceback
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from slalom_explanations import SLALOMLocalExplanantions
from xai_metrics import (
    calculate_log_odds_qa,
    calculate_soft_comprehensiveness_qa,
    calculate_soft_sufficiency_qa,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# QA → classification adapter for SLALOM
# ---------------------------------------------------------------------------

class QAModelWrapper(torch.nn.Module):
    """
    Wraps an AutoModelForQuestionAnswering so it presents a classification-
    compatible interface that SLALOM accepts.

    SLALOM calls model(input_ids) and expects a plain logit Tensor.
    QA models return a QuestionAnsweringModelOutput, which SLALOM rejects.

    Strategy chosen: expose start_logits as the output logits.
    SLALOM will then compute token attributions w.r.t. the start-position
    distribution, which is a reasonable single proxy for the QA task
    (start and end attributions are typically highly correlated for
    extractive QA, and SLALOM cannot target two heads simultaneously).

    The wrapper also stores the last start_logits / end_logits so that
    the outer code can read true QA predictions after a forward pass.
    """

    def __init__(self, qa_model):
        super().__init__()
        self.qa_model      = qa_model
        self.last_start_logits = None
        self.last_end_logits   = None

    # Expose the underlying embeddings so SLALOM's internals can reach them.
    def get_input_embeddings(self):
        return self.qa_model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None, **kwargs):
        out = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        self.last_start_logits = out.start_logits   # [B, L]
        self.last_end_logits   = out.end_logits     # [B, L]
        # Return start_logits as the "classification logits" SLALOM expects.
        # Shape [B, L] — SLALOM treats each position as a "class".
        return out.start_logits


# ---------------------------------------------------------------------------
# Output structure detection  (shared logic with slalom_eval.py)
# ---------------------------------------------------------------------------
_SLALOM_FORMAT = None   # detected lazily on first call


def _detect_and_unpack(res):
    """
    Auto-detect SLALOM output format and return (tokens, values, imps).
    See slalom_eval.py for the four format variants.
    """
    global _SLALOM_FORMAT

    if _SLALOM_FORMAT is None:
        if isinstance(res, dict):
            _SLALOM_FORMAT = "dict"
            print(f"[SLALOM format detected] dict, keys={list(res.keys())}")
        elif isinstance(res, (list, tuple)) and len(res) > 0:
            elem = res[0]
            if isinstance(elem, (list, tuple)):
                n = len(elem)
                if n >= 3:
                    _SLALOM_FORMAT = "list_3tuple"
                elif n == 2:
                    v = np.array(elem[1])
                    if v.ndim == 2:
                        _SLALOM_FORMAT = "list_2tuple_stacked"
                    else:
                        _SLALOM_FORMAT = "list_2tuple_single"
                else:
                    raise ValueError(f"Unexpected tuple length {n}: {elem}")
            else:
                raise ValueError(f"Unexpected element type {type(elem)}: {elem}")
        else:
            raise ValueError(f"Unexpected SLALOM output type {type(res)}: {res}")
        print(f"[SLALOM format] {_SLALOM_FORMAT}")

    def _to_1d(x):
        """(L,) or (L,C) → (L,) signed difference for binary, or first dim for scalar."""
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            return x
        elif x.shape[-1] == 1:
            return x[..., 0]
        else:
            return x[..., 1] - x[..., 0]

    if _SLALOM_FORMAT == "dict":
        tokens = res["tokens"]
        values = _to_1d(np.array(res["value"], dtype=np.float32))
        imps   = _to_1d(np.array(res["imp"],   dtype=np.float32))

    elif _SLALOM_FORMAT == "list_3tuple":
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = _to_1d(np.stack([np.array(r[2], dtype=np.float32) for r in res]))

    elif _SLALOM_FORMAT == "list_2tuple_stacked":
        tokens  = [r[0] for r in res]
        stacked = np.stack([np.array(r[1], dtype=np.float32) for r in res])
        values  = _to_1d(stacked[:, 0, :])
        imps    = _to_1d(stacked[:, 1, :]) if stacked.shape[1] > 1 else np.zeros(len(tokens), dtype=np.float32)

    elif _SLALOM_FORMAT == "list_2tuple_single":
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = np.zeros(len(tokens), dtype=np.float32)

    return tokens, values, imps


# ---------------------------------------------------------------------------
# QA-specific metric helpers
# ---------------------------------------------------------------------------

def _get_base_emb(model, tokenizer, device):
    mask_id  = tokenizer.mask_token_id or tokenizer.pad_token_id
    qa_model = _unwrap(model)
    with torch.no_grad():
        return qa_model.get_input_embeddings()(
            torch.tensor([[mask_id]], device=device)
        ).squeeze(0)


def _unwrap(model):
    """Return the underlying QA model from a QAModelWrapper, or model itself."""
    return model.qa_model if isinstance(model, QAModelWrapper) else model


def compute_soft_metrics_qa(
    model, tokenizer, device,
    input_ids, attention_mask, token_type_ids,
    attr_start, attr_end, topk=20, n_samples=10,
):
    """
    Compute log-odds + soft-comprehensiveness + soft-sufficiency for QA
    start and end positions using the xai_metrics soft QA functions.

    model may be a QAModelWrapper or a raw AutoModelForQuestionAnswering.
    attr_start / attr_end : [seq_len] tensors aligned to the full tokenized sequence.

    Returns:
        log_odd_start, log_odd_end,
        soft_comp_start, soft_comp_end,
        soft_suff_start, soft_suff_end,
        start_idx, end_idx
    """
    qa_model = _unwrap(model)
    embed = qa_model.get_input_embeddings()
    with torch.no_grad():
        X   = embed(input_ids)        # [1, seq, d]
        out = qa_model(
            inputs_embeds=X,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    start_idx = int(out.start_logits.argmax(dim=-1).item())
    end_idx   = int(out.end_logits.argmax(dim=-1).item())

    prob_start_orig = F.softmax(out.start_logits, dim=-1)[0, start_idx]
    prob_end_orig   = F.softmax(out.end_logits,   dim=-1)[0, end_idx]

    base_emb = _get_base_emb(model, tokenizer, device)  # (1, d)

    # Build special_tokens_mask for soft metric functions
    special_ids = set(tokenizer.all_special_ids)
    special_tokens_mask = torch.tensor(
        [tid in special_ids for tid in input_ids[0].tolist()],
        device=device, dtype=torch.bool,
    )

    # NOTE: soft QA functions operate on the RAW QA model, not the wrapper.
    log_odd_start, log_odd_end = calculate_log_odds_qa(
        qa_model, X, attention_mask, special_tokens_mask, token_type_ids,
        base_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, topk=topk,
    )
    soft_comp_start, soft_comp_end = calculate_soft_comprehensiveness_qa(
        qa_model, X, attention_mask, special_tokens_mask, token_type_ids,
        base_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )
    soft_suff_start, soft_suff_end = calculate_soft_sufficiency_qa(
        qa_model, X, attention_mask, special_tokens_mask, token_type_ids,
        base_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig, n_samples=n_samples,
    )

    return (
        log_odd_start, log_odd_end,
        soft_comp_start, soft_comp_end,
        soft_suff_start, soft_suff_end,
        start_idx, end_idx,
    )


# ---------------------------------------------------------------------------
# Single-example wrapper
# ---------------------------------------------------------------------------

def slalom_explain_and_eval_qa(
    question, context,
    qa_model, tokenizer,
    device, topk=20, attr_mode="lin", n_samples=10,
):
    """
    Run SLALOM on a (question, context) pair, build attribution vectors for
    start and end positions, then compute soft QA metrics.

    Creates a fresh SLALOM explainer per example because QA requires
    target_class = predicted start_idx (different per example).

    attr_mode:
        "value" — use SLALOM value scores directly
        "imp"   — use SLALOM importance scores
        "lin"   — linearized: value * exp(imp)   (paper default, Section B.7)
    """
    t0 = time.perf_counter()

    #  First forward pass to get predicted start_idx 
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = qa_model(**enc)
    start_idx = int(out.start_logits.argmax(dim=-1).item())

    #  Build SLALOM explainer with target_class = start_idx 
    wrapper = QAModelWrapper(qa_model)
    wrapper.eval()
    slalom_explainer = SLALOMLocalExplanantions(
        wrapper, tokenizer, modes=["value", "imp"], target_class=start_idx,
    )

    # Build the combined text that mirrors the QA tokenization.
    combined_text = question + " " + tokenizer.sep_token + " " + context

    raw = slalom_explainer.tokenize_and_explain(combined_text)
    t1  = time.perf_counter()

    tokens_out, values, imps = _detect_and_unpack(raw)

    # Build base attribution vector
    if attr_mode == "value":
        attr_np = values
    elif attr_mode == "imp":
        attr_np = imps
    else:   # "lin"
        attr_np = values * np.exp(np.clip(imps, -20, 20))

    # Tokenize with QA-style pair encoding for metric computation
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
    )
    enc            = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
    L              = input_ids.shape[1]

    # Align SLALOM attribution length to the full tokenized sequence.
    # SLALOM may strip special tokens; re-insert zeros at special positions.
    attr_tensor = torch.tensor(attr_np, dtype=torch.float32)
    if attr_tensor.shape[0] != L:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx        = [i for i, tid in enumerate(input_ids[0].tolist())
                           if tid not in special_ids_set]
        full_attr = torch.zeros(L, dtype=torch.float32)
        if len(keep_idx) == attr_tensor.shape[0]:
            full_attr[keep_idx] = attr_tensor
        attr_tensor = full_attr

    # Use the same attribution for start and end (single SLALOM pass)
    attr_start = attr_tensor
    attr_end   = attr_tensor

    (
        log_odd_start, log_odd_end,
        soft_comp_start, soft_comp_end,
        soft_suff_start, soft_suff_end,
        start_idx, end_idx,
    ) = compute_soft_metrics_qa(
        qa_model, tokenizer, device,
        input_ids, attention_mask, token_type_ids,
        attr_start, attr_end,
        topk=topk, n_samples=n_samples,
    )

    # Decode predicted answer span
    predicted_answer = tokenizer.decode(
        input_ids[0, max(start_idx, 0): max(end_idx, start_idx) + 1],
        skip_special_tokens=True,
    )

    return {
        "tokens":           tokens_out,
        "value":            values.tolist(),
        "imp":              imps.tolist(),
        "lin":              (values * np.exp(np.clip(imps, -20, 20))).tolist(),
        "predicted_answer": predicted_answer,
        "start_idx":        start_idx,
        "end_idx":          end_idx,
        "time":             t1 - t0,
        "log_odd_start":    log_odd_start,
        "soft_comp_start":  soft_comp_start,
        "soft_suff_start":  soft_suff_start,
        "log_odd_end":      log_odd_end,
        "soft_comp_end":    soft_comp_end,
        "soft_suff_end":    soft_suff_end,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args):
    # SLALOM is fundamentally incompatible with QA models:
    # QA outputs [B, L] logits (variable L), but SLALOM internally samples
    # masked inputs of different lengths, changing L.  The target_class
    # (start index) then goes out of bounds on these shorter sequences.
    # SLALOM requires fixed-size classification output.
    print("SLALOM does not support QA models (variable-length output).")
    print("Use sentiment classification scripts instead.")
    return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model_name}")
    print(f"Dataset    : {args.dataset}")
    print(f"SLALOM mode: {args.attr_mode}")
    print(f"Top-k      : {args.topk}%")
    print(f"Num samples: {args.num_samples}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    qa_model  = AutoModelForQuestionAnswering.from_pretrained(args.model_name).to(device)
    qa_model.eval()

    #  dataset 
    print(f"\nLoading dataset: {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="validation")
    data = list(zip(
        dataset["question"], dataset["context"],
        dataset["answers"],  dataset["id"],
    ))

    # Keep only short examples (same filter as run_eval_pg_qa.py)
    upd_data = [
        (q, c, a, idx)
        for q, c, a, idx in data
        if len((q + c).split()) < 80
    ]
    print(f"Filtered samples: {len(upd_data)}")

    answerable_data = [
        {"context": c, "question": q, "answers": a}
        for q, c, a, _ in upd_data
    ]

    if len(answerable_data) > args.num_samples:
        sampled_data = random.sample(answerable_data, args.num_samples)
    else:
        sampled_data = answerable_data
        print(f"Warning: Only {len(answerable_data)} samples available.")

    print(f"Evaluating {len(sampled_data)} samples with SLALOM QA ...\n")

    total_log_odd_start  = total_log_odd_end  = 0.0
    total_soft_comp_start = total_soft_comp_end = 0.0
    total_soft_suff_start = total_soft_suff_end = 0.0
    total_time = 0.0
    count = errors = 0

    for example in tqdm(sampled_data):
        try:
            res = slalom_explain_and_eval_qa(
                question=example["question"],
                context=example["context"],
                qa_model=qa_model,
                tokenizer=tokenizer,
                device=device,
                topk=args.topk,
                attr_mode=args.attr_mode,
                n_samples=args.n_samples,
            )
            total_log_odd_start  += res["log_odd_start"]
            total_log_odd_end    += res["log_odd_end"]
            total_soft_comp_start += res["soft_comp_start"]
            total_soft_comp_end   += res["soft_comp_end"]
            total_soft_suff_start += res["soft_suff_start"]
            total_soft_suff_end   += res["soft_suff_end"]
            total_time          += res["time"]
            count += 1

            if count % args.print_step == 0:
                print(f"\n[{count}/{len(sampled_data)}]"
                      f"  log-odds(s)={total_log_odd_start/count:.4f}"
                      f"  log-odds(e)={total_log_odd_end/count:.4f}"
                      f"  soft-comp(s)={total_soft_comp_start/count:.4f}"
                      f"  soft-comp(e)={total_soft_comp_end/count:.4f}"
                      f"  soft-suff(s)={total_soft_suff_start/count:.4f}"
                      f"  soft-suff(e)={total_soft_suff_end/count:.4f}"
                      f"  time={total_time/count:.4f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                traceback.print_exc()

    n = max(count, 1)
    print(f"\n{''*60}")
    print(f"SLALOM ({args.attr_mode})  |  {args.model_name} / {args.dataset}")
    print(f"  Log-odds (start):          {total_log_odd_start / n:.6f}")
    print(f"  Soft-Comprehensiveness (start): {total_soft_comp_start / n:.6f}")
    print(f"  Soft-Sufficiency (start):       {total_soft_suff_start / n:.6f}")
    print(f"  Log-odds (end):            {total_log_odd_end   / n:.6f}")
    print(f"  Soft-Comprehensiveness (end):   {total_soft_comp_end / n:.6f}")
    print(f"  Soft-Sufficiency (end):         {total_soft_suff_end / n:.6f}")
    print(f"  Avg time/sample:           {total_time / n:.4f}s")
    print(f"  Evaluated: {count}  |  Errors: {errors}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SLALOM explanations on QA datasets."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="deepset/bert-base-cased-squad2",
        help="HuggingFace QA model name",
    )
    parser.add_argument(
        "--dataset", type=str, default="squad",
        choices=["squad", "squad_v2"],
        help="HuggingFace dataset to evaluate on",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Percentage of top tokens for metrics calculation",
    )
    parser.add_argument(
        "--attr_mode", choices=["value", "imp", "lin"], default="lin",
        help="Attribution mode: value | imp | lin (default: lin)",
    )
    parser.add_argument(
        "--print_step", type=int, default=100,
        help="Print running averages every N samples",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Stochastic samples for soft metrics",
    )

    args = parser.parse_args()
    run_benchmark(args)