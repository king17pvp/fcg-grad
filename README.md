# FCGGrad - Functional Consistent Gated Gradient

Token-level attribution method for Transformer models. FCGGrad replaces step-wise delta weighting with a scalar gating mechanism that measures how much each token's embedding change contributes to the model output change along the integration path.

Supports **sentiment classification** (DistilBERT, BERT, RoBERTa on SST-2, IMDB, Rotten Tomatoes) and **extractive QA** (SQuAD).

## Setup

**Requirements:** Python 3.12, CUDA-compatible GPU (or CPU-only with `torch` CPU build).

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate fcg_gradient_env
```

### Option B: pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify

```bash
python -c "import torch; import transformers; import captum; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```

## Quick start

```bash
# Sentiment Analysis on SST-2 with DistilBERT
python run_eval_fcg_sentiment.py --model distilbert --dataset sst2

# QA on SQuAD with BERT
python run_eval_fcg_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 500
```

## Sentiment classification

```bash
python run_eval_fcg_sentiment.py \
  --model {distilbert,bert,roberta} \
  --dataset {sst2,imdb,rotten} \
  --steps 100 \
  --baseline mask \
  --eval-baseline mask \
  --n-samples 10
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `distilbert` | Model architecture |
| `--dataset` | (required) | `sst2`, `imdb`, or `rotten` |
| `--steps` | `100` | Integration steps along `[0, 1]` |
| `--baseline` | `mask` | Reference embedding for the integration path: `mask`, `pad`, `zero`, `mean`, `random` |
| `--eval-baseline` | `mask` | Embedding used to ablate tokens in faithfulness metrics |
| `--n-samples` | `10` | Bernoulli samples for soft metrics |

**Metrics reported:** Log-Odds, Soft-Comprehensiveness, Soft-Sufficiency (Zhao & Aletras, ACL 2023).

## Extractive QA

```bash
python run_eval_fcg_qa.py \
  --model_name deepset/bert-base-cased-squad2 \
  --dataset squad \
  --steps 100 \
  --baseline mask \
  --eval-baseline mask \
  --num_samples 1000 \
  --topk 50 \
  --n-samples 10
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | `cleandata/bert-finetuned-squad` | HuggingFace QA model |
| `--dataset` | `squad` | Dataset name |
| `--steps` | `100` | Integration steps |
| `--baseline` | `mask` | Reference embedding for integration path |
| `--eval-baseline` | `mask` | Embedding for metric ablation |
| `--num_samples` | `1000` | Max evaluation samples |
| `--topk` | `50` | Top-k% tokens for log-odds |
| `--n-samples` | `10` | Bernoulli samples for soft metrics |

**Metrics reported:** Log-Odds, Soft-Comprehensiveness, Soft-Sufficiency, computed separately for start and end answer positions.

## Visualization

```bash
# Sentiment Analysis FCGGrad vs IG comparison
python visualization_fcg.py --model distilbert --dataset sst2 --num_samples 30

# QA FCGGrad vs IG comparison
python visualization_fcg_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 30
```

Output saved to `visualizations/` and `visualizations_qa/` respectively.

## Ablations

Gradient normalisation and gate ablation scripts live in `ablations/`:

```bash
# Gradient normalisation variants
python ablations/run_eval_fcg_sentiment_gnorm.py --gnorm sign_norm --model distilbert --dataset sst2

# Gate ablation (L1 / L2 / scalar)
python ablations/run_eval_fcg_sentiment_ablation.py --method l2 --model bert --dataset imdb

# Reference embedding ablation
python ablations/run_ablation_fcg_ref_embed.py --models bert --datasets sst2
```

## Experiments

Other attribution methods (IG, AttCAT, ReAGent, FI, SLALOM, Flexy, GAF) live in `experiments/`:

```bash
python experiments/run_eval_ig_sentiment.py --model distilbert --dataset sst2
python experiments/run_eval_attcat_qa.py --model_name deepset/bert-base-cased-squad2
```

## Repository structure

```
FCGGrad/
├── fcg_gradients.py              # Core FCG implementation
├── xai_metrics.py                # Faithfulness metrics (hard + soft)
├── vanilla_ig.py                 # Integrated Gradients
├── run_eval_fcg_sentiment.py     # Sentiment eval
├── run_eval_fcg_qa.py            # QA eval
├── visualization_fcg.py          # Sentiment viz (FCG vs IG)
├── visualization_fcg_qa.py       # QA viz (FCG vs IG)
├── helpers/                      # Per-model forward helpers
│   ├── bert_helper.py
│   ├── distilbert_helper.py
│   └── roberta_helper.py
├── ablations/                    # Ablation studies
│   ├── fcg_gradient_gnorm.py
│   ├── fcg_gradient_ablations.py
│   └── run_eval_fcg_sentiment_*.py
├── experiments/                  # Non-FCG methods
│   ├── run_eval_ig_*.py
│   ├── run_eval_attcat_*.py
│   ├── run_eval_reagent_*.py
│   ├── run_eval_fi_*.py
│   └── ...
└── scripts/                      # Batch runners
    ├── run_fcg_ablation_all.sh
    └── run_fcg_gnorm_all.sh
```
