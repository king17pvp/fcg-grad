"""
XAI Metrics for Attribution Evaluation

This module contains metrics for evaluating token-level attributions:
- Log-odds: Change in log probability when important tokens are masked
- Comprehensiveness: Probability drop when important tokens are removed
- Sufficiency: How well top-k tokens alone preserve the prediction

Includes both classification metrics and QA-specific metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# CLASSIFICATION

def calculate_log_odds(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate log-odds for classification tasks."""
	logits_original						= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label						= torch.argmax(logits_original).item()
	prob_original						= torch.softmax(logits_original, dim=0)
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= base_token_emb
	logits_perturbed					= foward_func(model, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed						= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), predicted_label


def calculate_sufficiency(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate sufficiency for classification tasks."""
	logits_original							= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label							= torch.argmax(logits_original).item()
	prob_original							= torch.softmax(logits_original, dim=0)
	topk_indices							= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	if len(topk_indices) == 0:
		# topk% is too less to select even word - so no masking will happen.
		return 0

	mask									= torch.zeros_like(input_embed[0][:,0]).bool()
	mask[topk_indices]						= 1
	masked_input_embed						= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask					= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed					= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed						= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed						= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed							= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


def calculate_comprehensiveness(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate comprehensiveness for classification tasks."""
	logits_original					= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label					= torch.argmax(logits_original).item()
	prob_original					= torch.softmax(logits_original, dim=0)
	topk_indices					= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	mask 							= torch.ones_like(input_embed[0][:,0]).bool()
	mask[topk_indices]				= 0
	masked_input_embed				= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask			= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed			= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed				= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed				= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed					= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


# QUESTION ANSWERING

def _to_tensor_scalar(x, device):
    """Ensure x is a 0-dim float Tensor on the correct device (accepts Tensor or float)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(float(x), dtype=torch.float32, device=device)


def calculate_log_odds_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids, 
                          base_token_emb, attr_start, attr_end, start_idx, end_idx,
                          prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate log-odds metric for Question Answering.
    
    Measures the change in log probability when top-k attributed tokens are masked.
    Uses SEPARATE attributions for start and end positions - no combined scores.
    More negative values indicate that important tokens were correctly identified.
    
    Args:
        model: QA model (AutoModelForQuestionAnswering)
        input_embed: Original input embeddings (1, L, d)
        attention_mask: Attention mask (1, L)
        special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be masked
        token_type_ids: Token type IDs (1, L) or None
        base_token_emb: Baseline token embedding for masking (1, d)
        attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
        attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
        start_idx: Predicted answer start index
        end_idx: Predicted answer end index
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)
        topk: Percentage of top tokens to mask (default: 50%)
    
    Returns:
        Tuple of (log_odds_start, log_odds_end): Log odds difference for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs so torch.log always receives a Tensor
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids
     
    num_tokens = attr_start.shape[0]
    # Count maskable tokens (excluding special tokens)
    num_maskable = num_tokens - special_token_mask.sum().item()
    k = max(1, int(num_maskable * topk / 100))
    
    # ===== Compute log_odds for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, k, sorted=False).indices
    
    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]
    
    log_odds_start = (torch.log(prob_start_pert + 1e-10) - torch.log(prob_start_orig + 1e-10)).item()
    
    # ===== Compute log_odds for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, k, sorted=False).indices
    
    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]
    
    log_odds_end = (torch.log(prob_end_pert + 1e-10) - torch.log(prob_end_orig + 1e-10)).item()
    
    return log_odds_start, log_odds_end


def calculate_comprehensiveness_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
                                   base_token_emb, attr_start, attr_end, start_idx, end_idx,
                                   prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate comprehensiveness metric for Question Answering.
    
    Measures probability drop when top-k attributed tokens are removed.
    Uses SEPARATE attributions for start and end positions - no combined scores.
    Higher values indicate that important tokens were correctly identified.
    
    Args:
        model: QA model
        input_embed: Original input embeddings (1, L, d)
        attention_mask: Attention mask (1, L)
        special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be removed
        token_type_ids: Token type IDs (1, L) or None
        base_token_emb: Baseline token embedding (1, d)
        attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
        attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
        start_idx: Predicted answer start index
        end_idx: Predicted answer end index
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)
        topk: Percentage of top tokens to remove (default: 50%)
    
    Returns:
        Tuple of (comp_start, comp_end): Probability drop for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids
    
    # ===== Compute comprehensiveness for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices
    
    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        new_len_start = start_logits_pert.shape[0]
        new_start_idx = min(start_idx, new_len_start - 1)
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[new_start_idx]
    
    comp_start = (prob_start_orig - prob_start_pert).item()
    
    # ===== Compute comprehensiveness for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices
    
    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        new_len_end = end_logits_pert.shape[0]
        new_end_idx = min(end_idx, new_len_end - 1)
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[new_end_idx]
    
    comp_end = (prob_end_orig - prob_end_pert).item()
    
    return comp_start, comp_end


def calculate_sufficiency_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
                             base_token_emb, attr_start, attr_end, start_idx, end_idx,
                             prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate sufficiency metric for Question Answering (MASKING, not deletion).

    Keeps only the top-k attributed tokens (plus special tokens) by masking all other tokens
    with base_token_emb while keeping sequence length fixed.

    Args:
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)

    Returns:
        Tuple of (suff_start, suff_end): Probability drop for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    num_tokens = attr_start.shape[0]
    num_maskable = num_tokens - special_token_mask.sum().item()
    k = max(1, int(num_maskable * topk / 100))

    # ===== Compute sufficiency for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices

    keep_mask_start = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    keep_mask_start[topk_indices_start] = True
    keep_mask_start[special_token_mask] = True

    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0, ~keep_mask_start, :] = base_token_emb

    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]

    suff_start = (prob_start_orig - prob_start_pert).item()

    # ===== Compute sufficiency for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices

    keep_mask_end = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    keep_mask_end[topk_indices_end] = True
    keep_mask_end[special_token_mask] = True

    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0, ~keep_mask_end, :] = base_token_emb

    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]

    suff_end = (prob_end_orig - prob_end_pert).item()

    return suff_start, suff_end


def eval_wae(scaled_features, word_embedding, epsilon=0.1):
	"""
	Compute Word Alignment Error (WAE).
	
	Measures the smallest distance of each embedding to any word embedding
	and reports the average among all words in the path for a sentence.
	"""
	dists = []
	for emb in scaled_features:
		all_dist = torch.sqrt(torch.sum((word_embedding - emb.unsqueeze(0)) ** 2, dim=1))
		dists.append(torch.min(all_dist).item())

	return np.mean(dists)


# SOFT FAITHFULNESS METRICS
# Zhao & Aletras, "Incorporating Attribution Importance for Improving Faithfulness Metrics", ACL 2023.
# https://aclanthology.org/2023.acl-long.261
#
# Instead of hard top-k erasure, these metrics apply a Bernoulli dropout mask
# to token embedding dimensions, with dropout probability proportional to the
# FA attribution score.  This avoids the hard rationale-length hyperparameter
# and reduces out-of-distribution artifacts.
#
# Soft-NS  (Eq. 4): sufficiency mode       : retain ∝ importance  (q = a_i)
# Soft-NC  (Eq. 5): comprehensiveness mode : erase  ∝ importance  (q = 1 - a_i)


def _normalize_attr_scores_soft(attr: torch.Tensor) -> torch.Tensor:
    """Min-max normalize attribution scores to [0, 1] for use as Bernoulli probabilities.

    If the range is effectively zero (all scores equal), returns 0.5 for every token.
    """
    a_min = attr.min()
    a_max = attr.max()
    if (a_max - a_min).abs() < 1e-9:
        return torch.full_like(attr, 0.5)
    return (attr - a_min) / (a_max - a_min)


def _soft_perturb_embed(
    input_embed: torch.Tensor,
    attr: torch.Tensor,
    mode: str,
    special_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply per-token Bernoulli dropout to token embedding dimensions (Eq. 3).

    Parameters
    ----------
    input_embed       : (1, seq_len, d) — token embeddings to perturb
    attr              : (seq_len,)      — raw FA attribution scores
    mode              : "sufficiency"       -> q_i = a_i   (retain ∝ importance)
                        "comprehensiveness" -> q_i = 1-a_i (erase  ∝ importance)
    special_token_mask: (seq_len,) bool, optional — positions that should never
                        be dropped (CLS, SEP, PAD); their q is forced to 1.

    Returns
    -------
    Perturbed embeddings of shape (1, seq_len, d).  Masked dimensions are set to 0.
    """
    assert mode in ("sufficiency", "comprehensiveness"), \
        f"mode must be 'sufficiency' or 'comprehensiveness', got '{mode}'"

    scores = _normalize_attr_scores_soft(attr.float())           # (seq_len,) in [0,1]
    q      = scores if mode == "sufficiency" else (1.0 - scores)

    if special_token_mask is not None:
        q = q.clone()
        q[special_token_mask] = 1.0  # always retain special tokens

    device = input_embed.device
    mask   = torch.bernoulli(q.to(device))           # (seq_len,) in {0, 1}
    mask   = mask.unsqueeze(0).unsqueeze(-1)          # (1, seq_len, 1) broadcasts over d
    return input_embed.detach() * mask


def calculate_soft_sufficiency(
    foward_func,
    model,
    input_embed:     torch.Tensor,
    position_embed:  torch.Tensor,
    type_embed,
    attention_mask:  torch.Tensor,
    base_token_emb:  torch.Tensor,
    attr:            torch.Tensor,
    n_samples:       int = 10,
) -> float:
    """Soft Normalized Sufficiency (Soft-NS) for classification — Eq. 4, Zhao & Aletras ACL 2023.

    Soft-S  = 1 - max(0, p(ŷ|X) - p(ŷ|X'))
    Soft-NS = (Soft-S - S(X,ŷ,0)) / (1 - S(X,ŷ,0))

    X' is produced by Bernoulli(a_i) elementwise dropout on word embeddings,
    using the full attribution distribution (no top-k cutoff).
    The result is averaged over n_samples stochastic draws to reduce variance.

    Parameters
    ----------
    foward_func    : bert_helper / distilbert_helper / roberta_helper nn_forward_func
    model          : fine-tuned sequence classifier
    input_embed    : (1, seq_len, d) — word embedding component only
    position_embed : (1, seq_len, d) — added internally by foward_func
    type_embed     : (1, seq_len, d) or None — added internally by foward_func
    attention_mask : (1, seq_len)
    base_token_emb : (1, d) — baseline word embedding (e.g. [MASK] token)
    attr           : (seq_len,) — FA attribution scores (any real values)
    n_samples      : number of Bernoulli samples to average

    Returns
    -------
    Soft-NS score (float).  Higher = rationale is more sufficient.
    """
    with torch.no_grad():
        logits_full = foward_func(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    predicted_label = torch.argmax(logits_full).item()
    p_full          = torch.softmax(logits_full, dim=0)[predicted_label].item()

    # S(X, ŷ, 0) — sufficiency of the all-baseline word embedding sequence
    seq_len    = input_embed.shape[1]
    zero_embed = base_token_emb.unsqueeze(0).expand(1, seq_len, -1).to(input_embed.device)
    with torch.no_grad():
        logits_base = foward_func(
            model, zero_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    p_base = torch.softmax(logits_base, dim=0)[predicted_label].item()
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom  = 1.0 - s_base   # = max(0, p_full - p_base)
    if abs(denom) < 1e-9:
        return 0.0

    soft_s_vals = []
    for _ in range(n_samples):
        x_prime = _soft_perturb_embed(input_embed, attr, mode="sufficiency")
        with torch.no_grad():
            logits_prime = foward_func(
                model, x_prime,
                attention_mask=attention_mask,
                position_embed=position_embed,
                type_embed=type_embed,
                return_all_logits=True,
            ).squeeze()
        p_prime = torch.softmax(logits_prime, dim=0)[predicted_label].item()
        soft_s_vals.append(1.0 - max(0.0, p_full - p_prime))

    soft_s = float(np.mean(soft_s_vals))
    return (soft_s - s_base) / denom


def calculate_soft_comprehensiveness(
    foward_func,
    model,
    input_embed:     torch.Tensor,
    position_embed:  torch.Tensor,
    type_embed,
    attention_mask:  torch.Tensor,
    base_token_emb:  torch.Tensor,
    attr:            torch.Tensor,
    n_samples:       int = 10,
) -> float:
    """Soft Normalized Comprehensiveness (Soft-NC) for classification — Eq. 5, Zhao & Aletras ACL 2023.

    Soft-C  = max(0, p(ŷ|X) - p(ŷ|X'))
    Soft-NC = Soft-C / (1 - S(X,ŷ,0))

    X' is produced by Bernoulli(1-a_i) elementwise dropout on word embeddings.
    More important tokens are more heavily erased, using the full attribution
    distribution without any hard top-k selection.
    The result is averaged over n_samples stochastic draws to reduce variance.

    Parameters
    ----------
    foward_func    : bert_helper / distilbert_helper / roberta_helper nn_forward_func
    model          : fine-tuned sequence classifier
    input_embed    : (1, seq_len, d) — word embedding component only
    position_embed : (1, seq_len, d) — added internally by foward_func
    type_embed     : (1, seq_len, d) or None — added internally by foward_func
    attention_mask : (1, seq_len)
    base_token_emb : (1, d) — baseline word embedding (e.g. [MASK] token)
    attr           : (seq_len,) — FA attribution scores (any real values)
    n_samples      : number of Bernoulli samples to average

    Returns
    -------
    Soft-NC score (float).  Higher = rationale is more comprehensive.
    """
    with torch.no_grad():
        logits_full = foward_func(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    predicted_label = torch.argmax(logits_full).item()
    p_full          = torch.softmax(logits_full, dim=0)[predicted_label].item()

    # S(X, ŷ, 0) — sufficiency of the all-baseline word embedding sequence
    seq_len    = input_embed.shape[1]
    zero_embed = base_token_emb.unsqueeze(0).expand(1, seq_len, -1).to(input_embed.device)
    with torch.no_grad():
        logits_base = foward_func(
            model, zero_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    p_base = torch.softmax(logits_base, dim=0)[predicted_label].item()
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom  = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0

    soft_c_vals = []
    for _ in range(n_samples):
        x_prime = _soft_perturb_embed(input_embed, attr, mode="comprehensiveness")
        with torch.no_grad():
            logits_prime = foward_func(
                model, x_prime,
                attention_mask=attention_mask,
                position_embed=position_embed,
                type_embed=type_embed,
                return_all_logits=True,
            ).squeeze()
        p_prime = torch.softmax(logits_prime, dim=0)[predicted_label].item()
        soft_c_vals.append(max(0.0, p_full - p_prime))

    soft_c = float(np.mean(soft_c_vals))
    return soft_c / denom

def calculate_soft_sufficiency_qa(
    model,
    input_embed:        torch.Tensor,
    attention_mask:     torch.Tensor,
    special_token_mask: torch.Tensor,
    token_type_ids,
    base_token_emb:     torch.Tensor,
    attr_start:         torch.Tensor,
    attr_end:           torch.Tensor,
    start_idx:          int,
    end_idx:            int,
    prob_start_orig,
    prob_end_orig,
    n_samples:          int = 10,
) -> Tuple[float, float]:
    """Soft Normalized Sufficiency (Soft-NS) for Question Answering.

    Extends the classification Soft-NS to extractive QA by computing separate
    metrics for the start and end answer positions, each driven by its own
    attribution map (attr_start / attr_end).

    Soft-S_start  = 1 - max(0, p(start|X) - p(start|X'_start))
    Soft-NS_start = (Soft-S_start - S_base_start) / (1 - S_base_start)
    (analogously for end)

    Special tokens (CLS, SEP, PAD) are never soft-perturbed — their Bernoulli
    probability is forced to 1 (always retain).

    Parameters
    ----------
    model               : AutoModelForQuestionAnswering
    input_embed         : (1, seq_len, d) — full embeddings (word+pos+type)
    attention_mask      : (1, seq_len)
    special_token_mask  : (seq_len,) bool — True = CLS/SEP/PAD, never perturbed
    token_type_ids      : (1, seq_len) or None
    base_token_emb      : (1, d) — baseline full embedding for non-special tokens
    attr_start          : (seq_len,) — FA scores for start logit
    attr_end            : (seq_len,) — FA scores for end logit
    start_idx           : predicted answer start position
    end_idx             : predicted answer end position
    prob_start_orig     : p(start_idx | X) — scalar Tensor or float
    prob_end_orig       : p(end_idx   | X) — scalar Tensor or float
    n_samples           : number of Bernoulli samples to average

    Returns
    -------
    Tuple (soft_ns_start, soft_ns_end).  Higher = more sufficient.
    """
    device          = input_embed.device
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    p_start_full = prob_start_orig.item()
    p_end_full   = prob_end_orig.item()

    # S(X, ŷ, 0): replace all non-special token positions with base_token_emb
    zero_embed = input_embed.detach().clone()
    zero_embed[0, ~special_token_mask, :] = base_token_emb
    with torch.no_grad():
        out_base     = model(inputs_embeds=zero_embed, attention_mask=attention_mask, **extra_kwargs)
        p_base_start = F.softmax(out_base.start_logits[0], dim=0)[start_idx].item()
        p_base_end   = F.softmax(out_base.end_logits[0],   dim=0)[end_idx].item()

    s_base_start = 1.0 - max(0.0, p_start_full - p_base_start)
    s_base_end   = 1.0 - max(0.0, p_end_full   - p_base_end)
    denom_start  = 1.0 - s_base_start
    denom_end    = 1.0 - s_base_end

    soft_s_start_vals: list = []
    soft_s_end_vals:   list = []
    for _ in range(n_samples):
        x_prime_start = _soft_perturb_embed(
            input_embed, attr_start, mode="sufficiency",
            special_token_mask=special_token_mask,
        )
        x_prime_end = _soft_perturb_embed(
            input_embed, attr_end, mode="sufficiency",
            special_token_mask=special_token_mask,
        )
        with torch.no_grad():
            out_s         = model(inputs_embeds=x_prime_start, attention_mask=attention_mask, **extra_kwargs)
            p_prime_start = F.softmax(out_s.start_logits[0], dim=0)[start_idx].item()
            out_e         = model(inputs_embeds=x_prime_end,   attention_mask=attention_mask, **extra_kwargs)
            p_prime_end   = F.softmax(out_e.end_logits[0],   dim=0)[end_idx].item()

        soft_s_start_vals.append(1.0 - max(0.0, p_start_full - p_prime_start))
        soft_s_end_vals.append(  1.0 - max(0.0, p_end_full   - p_prime_end))

    soft_s_start = float(np.mean(soft_s_start_vals))
    soft_s_end   = float(np.mean(soft_s_end_vals))

    result_start = (soft_s_start - s_base_start) / denom_start if abs(denom_start) >= 1e-9 else 0.0
    result_end   = (soft_s_end   - s_base_end)   / denom_end   if abs(denom_end)   >= 1e-9 else 0.0
    return result_start, result_end


def calculate_soft_comprehensiveness_qa(
    model,
    input_embed:        torch.Tensor,
    attention_mask:     torch.Tensor,
    special_token_mask: torch.Tensor,
    token_type_ids,
    base_token_emb:     torch.Tensor,
    attr_start:         torch.Tensor,
    attr_end:           torch.Tensor,
    start_idx:          int,
    end_idx:            int,
    prob_start_orig,
    prob_end_orig,
    n_samples:          int = 10,
) -> Tuple[float, float]:
    """Soft Normalized Comprehensiveness (Soft-NC) for Question Answering.

    Extends the classification Soft-NC to extractive QA by computing separate
    metrics for the start and end answer positions, each driven by its own
    attribution map (attr_start / attr_end).

    Soft-C_start  = max(0, p(start|X) - p(start|X'_start))
    Soft-NC_start = Soft-C_start / (1 - S_base_start)
    (analogously for end)

    Special tokens (CLS, SEP, PAD) are never soft-perturbed — their Bernoulli
    probability is forced to 1 (always retain).

    Parameters
    ----------
    model               : AutoModelForQuestionAnswering
    input_embed         : (1, seq_len, d) — full embeddings (word+pos+type)
    attention_mask      : (1, seq_len)
    special_token_mask  : (seq_len,) bool — True = CLS/SEP/PAD, never perturbed
    token_type_ids      : (1, seq_len) or None
    base_token_emb      : (1, d) — baseline full embedding for non-special tokens
    attr_start          : (seq_len,) — FA scores for start logit
    attr_end            : (seq_len,) — FA scores for end logit
    start_idx           : predicted answer start position
    end_idx             : predicted answer end position
    prob_start_orig     : p(start_idx | X) — scalar Tensor or float
    prob_end_orig       : p(end_idx   | X) — scalar Tensor or float
    n_samples           : number of Bernoulli samples to average

    Returns
    -------
    Tuple (soft_nc_start, soft_nc_end).  Higher = more comprehensive.
    """
    device          = input_embed.device
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    p_start_full = prob_start_orig.item()
    p_end_full   = prob_end_orig.item()

    # S(X, ŷ, 0): replace all non-special token positions with base_token_emb
    zero_embed = input_embed.detach().clone()
    zero_embed[0, ~special_token_mask, :] = base_token_emb
    with torch.no_grad():
        out_base     = model(inputs_embeds=zero_embed, attention_mask=attention_mask, **extra_kwargs)
        p_base_start = F.softmax(out_base.start_logits[0], dim=0)[start_idx].item()
        p_base_end   = F.softmax(out_base.end_logits[0],   dim=0)[end_idx].item()

    s_base_start = 1.0 - max(0.0, p_start_full - p_base_start)
    s_base_end   = 1.0 - max(0.0, p_end_full   - p_base_end)
    denom_start  = 1.0 - s_base_start
    denom_end    = 1.0 - s_base_end

    soft_c_start_vals: list = []
    soft_c_end_vals:   list = []
    for _ in range(n_samples):
        x_prime_start = _soft_perturb_embed(
            input_embed, attr_start, mode="comprehensiveness",
            special_token_mask=special_token_mask,
        )
        x_prime_end = _soft_perturb_embed(
            input_embed, attr_end, mode="comprehensiveness",
            special_token_mask=special_token_mask,
        )
        with torch.no_grad():
            out_s         = model(inputs_embeds=x_prime_start, attention_mask=attention_mask, **extra_kwargs)
            p_prime_start = F.softmax(out_s.start_logits[0], dim=0)[start_idx].item()
            out_e         = model(inputs_embeds=x_prime_end,   attention_mask=attention_mask, **extra_kwargs)
            p_prime_end   = F.softmax(out_e.end_logits[0],   dim=0)[end_idx].item()

        soft_c_start_vals.append(max(0.0, p_start_full - p_prime_start))
        soft_c_end_vals.append(  max(0.0, p_end_full   - p_prime_end))

    soft_c_start = float(np.mean(soft_c_start_vals))
    soft_c_end   = float(np.mean(soft_c_end_vals))

    result_start = soft_c_start / denom_start if abs(denom_start) >= 1e-9 else 0.0
    result_end   = soft_c_end   / denom_end   if abs(denom_end)   >= 1e-9 else 0.0
    return result_start, result_end


def calculate_soft_log_odds(
    foward_func,
    model,
    input_embed:     torch.Tensor,
    position_embed:  torch.Tensor,
    type_embed,
    attention_mask:  torch.Tensor,
    base_token_emb:  torch.Tensor,
    attr:            torch.Tensor,
    n_samples:       int = 10,
) -> float:
    """Soft log-odds via Bernoulli dropout (same perturbation as soft-comp).

    More negative = better attribution quality.
    """
    with torch.no_grad():
        logits_full = foward_func(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    pred_class = int(torch.softmax(logits_full, dim=-1).argmax().item())
    p_full = float(torch.softmax(logits_full, dim=-1)[pred_class].item())

    eps = 1e-9
    vals = []
    for _ in range(n_samples):
        x_prime = _soft_perturb_embed(input_embed, attr, mode="comprehensiveness")
        with torch.no_grad():
            logits_prime = foward_func(
                model, x_prime,
                attention_mask=attention_mask,
                position_embed=position_embed,
                type_embed=type_embed,
                return_all_logits=True,
            ).squeeze()
        p_prime = float(torch.softmax(logits_prime, dim=-1)[pred_class].item())
        lo = (np.log((p_full + eps) / (1 - p_full + eps))
            - np.log((p_prime + eps) / (1 - p_prime + eps)))
        vals.append(lo)

    return float(np.mean(vals))
