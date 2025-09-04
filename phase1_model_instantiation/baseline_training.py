"""
baseline_training.py
Purpose: prepare evaluation inputs for baseline runs (no metrics here).

Public functions:
- load_qa_jsonl(path) -> List[{"question": str, "answer": str}]
- build_eval_data_auto(tokenizer, processed_data_path, qa_jsonl_path, batch_size, max_domain_texts=None)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import sys

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]   
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from data.data_loader import QwenDataLoader
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import 'data.data_loader.QwenDataLoader'. "
        "Ensure the repository root is on sys.path and that 'data/' exists at the repo root."
    ) from e

# ---------------------------
# QA loader
# ---------------------------
def load_qa_jsonl(qa_jsonl_path: str) -> List[Dict[str, str]]:
    """
    Load held-out QA pairs from JSONL with keys: 'question', 'answer'.
    Use a small, curated file to avoid leakage from the DAPT corpus.
    """
    p = Path(qa_jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"QA eval file not found: {qa_jsonl_path}")

    qa_pairs: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                qa_pairs.append({"question": obj["question"], "answer": obj["answer"]})

    if not qa_pairs:
        raise ValueError(f"No QA pairs loaded from {qa_jsonl_path}")

    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_jsonl_path}")
    return qa_pairs


# ---------------------------
# Domain-text builder
# ---------------------------
def build_eval_data_auto(
    tokenizer,
    processed_data_path: str = "data/qwen_processed_data.pkl",
    qa_jsonl_path: str = "data/qa_eval.jsonl",
    batch_size: int = 8,
    max_domain_texts: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    # make relative paths resolve from repo root
    p_proc = Path(processed_data_path)
    if not p_proc.is_absolute():
        p_proc = _REPO_ROOT / p_proc
    processed_data_path = str(p_proc)

    p_qa = Path(qa_jsonl_path)
    if not p_qa.is_absolute():
        p_qa = _REPO_ROOT / p_qa
    qa_jsonl_path = str(p_qa)

    # now import/construct loader
    loader = QwenDataLoader(
        processed_data_path=processed_data_path,
        train_split=0.8, val_split=0.1, test_split=0.1,
        batch_size=batch_size,
    )
    test_loader = loader.get_test_dataloader()

    qa_pairs: List[Dict[str, str]] = []
    domain_texts: List[str] = []

    for batch in test_loader:
        # Optional QA if present
        if "question" in batch and "answer" in batch:
            qs, ans = batch["question"], batch["answer"]
            if isinstance(qs, str):
                qs = [qs]
            if isinstance(ans, str):
                ans = [ans]
            qa_pairs.extend({"question": q, "answer": a} for q, a in zip(qs, ans))

        # Domain texts: prefer raw text; else decode input_ids
        if "text" in batch:
            texts = batch["text"]
            if isinstance(texts, str):
                texts = [texts]
            domain_texts.extend(_normalize_text_list(texts))
        elif "input_ids" in batch:
            decoded = _safe_batch_decode_input_ids(batch["input_ids"], tokenizer)
            domain_texts.extend(decoded)

        # Optional cap for speed
        if max_domain_texts is not None and len(domain_texts) >= max_domain_texts:
            domain_texts = domain_texts[:max_domain_texts]
            break

    if not domain_texts:
        raise ValueError(
            "No domain texts from test_loader. Ensure batches provide 'text' or 'input_ids'."
        )

    if not qa_pairs:
        qa_pairs = load_qa_jsonl(qa_jsonl_path)

    logger.info(
        f"Prepared eval inputs → QA pairs: {len(qa_pairs)}, Domain texts: {len(domain_texts)}"
    )
    return qa_pairs, domain_texts


# ---------------------------
# Helpers
# ---------------------------
def _normalize_text_list(items) -> List[str]:
    # Ensure list[str], strip empties
    if isinstance(items, str):
        items = [items]
    out = []
    for t in items:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def _safe_batch_decode_input_ids(input_ids, tokenizer) -> List[str]:
    """
    Robustly decode input_ids that may come as tensors or lists.
    Uses the provided tokenizer (same as preprocessing tokenizer in your setup).
    """
    # torch.Tensor -> list on CPU; also handle list[Tensor]
    try:
        if hasattr(input_ids, "cpu"):
            input_ids = input_ids.cpu().tolist()
    except Exception:
        pass

    if isinstance(input_ids, list) and input_ids and hasattr(input_ids[0], "cpu"):
        input_ids = [ids.cpu().tolist() for ids in input_ids]

    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return [t.strip() for t in decoded if t and t.strip()]

# === Baseline evaluation helpers expected by run_all.py ===
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def _load_qa_jsonl(qa_jsonl_path: str) -> List[Dict[str, str]]:
    """Load held-out QA pairs from JSONL with keys: question, answer."""
    p = Path(qa_jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"QA eval file not found: {qa_jsonl_path}")
    qa_pairs: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                qa_pairs.append({"question": obj["question"], "answer": obj["answer"]})
    if not qa_pairs:
        raise ValueError(f"No QA pairs loaded from {qa_jsonl_path}")
    return qa_pairs

def build_eval_data_auto(
    tokenizer,
    processed_data_path: str = "data/qwen_processed_data.pkl",
    qa_jsonl_path: str = "data/qa_eval.jsonl",
    batch_size: int = 8,
    max_domain_texts: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Build (qa_pairs, domain_texts) for baseline evaluation.

    Logic:
      - Try to get QA directly from test_loader if it has 'question'/'answer'.
      - Always collect domain_texts from test_loader:
          * prefer 'text' field,
          * else decode 'input_ids' with tokenizer (common in DAPT corpora).
      - If no QA in loader (typical for DAPT), load a held-out QA JSONL.
    """
    from data.data_loader import QwenDataLoader

    loader = QwenDataLoader(
        processed_data_path=processed_data_path,
        train_split=0.8, val_split=0.1, test_split=0.1,
        batch_size=batch_size,
    )
    test_loader = loader.get_test_dataloader()

    qa_pairs: List[Dict[str, str]] = []
    domain_texts: List[str] = []

    for batch in test_loader:
        # 1) QA if available
        if "question" in batch and "answer" in batch:
            qs, ans = batch["question"], batch["answer"]
            if isinstance(qs, str): qs = [qs]
            if isinstance(ans, str): ans = [ans]
            qa_pairs.extend({"question": q, "answer": a} for q, a in zip(qs, ans))

        # 2) Domain texts from text or input_ids
        if "text" in batch:
            texts = batch["text"]
            if isinstance(texts, str): texts = [texts]
            domain_texts.extend(texts)
        elif "input_ids" in batch:
            input_ids = batch["input_ids"]
            try:
                decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            except Exception:
                decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            domain_texts.extend(decoded)

        if max_domain_texts is not None and len(domain_texts) >= max_domain_texts:
            domain_texts = domain_texts[:max_domain_texts]
            break

    # If no QA surfaced from loader, fall back to held-out file
    if not qa_pairs:
        qa_pairs = _load_qa_jsonl(qa_jsonl_path)

    if not domain_texts:
        raise ValueError(
            "No domain texts from test_loader; ensure batches yield 'text' or 'input_ids'."
        )

    return qa_pairs, domain_texts

# ---- Metrics core (EM/F1/Perplexity) ----
import re, string
from collections import Counter
import torch

def _normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)

def _f1_score_text(prediction, ground_truth):
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def _calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", max_length=max_length,
                truncation=True, padding=False
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)

def _generate_baseline_responses(model, tokenizer, questions, max_new_tokens=50, batch_size=4):
    device = next(model.parameters()).device
    all_responses = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            inputs = tokenizer(
                batch_questions, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                      # deterministic baseline
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # keep only the generated part (strip the prompt if echoed)
            for q, resp in zip(batch_questions, decoded):
                gen = resp[len(q):].strip() if resp.startswith(q) else resp.strip()
                all_responses.append(gen)
    return all_responses

def establish_baseline_performance(model, tokenizer, qa_pairs: List[Dict[str, str]], domain_texts: List[str]) -> Dict:
    """
    Compute EM/F1 over QA and perplexity over domain texts.
    Returns a dict used by run_all.
    """
    logger.info("Establishing baseline performance (pre-distribution shift)")
    questions = [qa["question"] for qa in qa_pairs]
    truths = [qa["answer"] for qa in qa_pairs]

    logger.info(f"Generating responses for {len(questions)} baseline questions")
    preds = _generate_baseline_responses(model, tokenizer, questions)

    exact_matches = [_exact_match_score(p, t) for p, t in zip(preds, truths)]
    f1_scores = [_f1_score_text(p, t) for p, t in zip(preds, truths)]

    qa_metrics = {
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "exact_match_std": float(np.std(exact_matches)) if exact_matches else 0.0,
        "f1_score_std": float(np.std(f1_scores)) if f1_scores else 0.0,
        "num_questions": len(questions),
    }

    logger.info(f"Calculating perplexity on {len(domain_texts)} domain texts")
    ppl = _calculate_perplexity(model, tokenizer, domain_texts)
    domain_metrics = {
        "perplexity": float(ppl),
        "log_perplexity": float(math.log(ppl)) if ppl > 0 else float("inf"),
        "num_texts": len(domain_texts),
    }

    results = {
        "model_state": "baseline_pre_distribution_shift",
        "qa_metrics": qa_metrics,
        "domain_metrics": domain_metrics,
        "evaluation_details": {
            "questions": questions,
            "predictions": preds,
            "ground_truths": truths,
            "individual_em": exact_matches,
            "individual_f1": f1_scores,
            "domain_texts": domain_texts,
        },
    }

    logger.info("Baseline Performance Results:")
    logger.info(f"  Exact Match: {qa_metrics['exact_match']:.3f} (±{qa_metrics['exact_match_std']:.3f})")
    logger.info(f"  F1 Score: {qa_metrics['f1_score']:.3f} (±{qa_metrics['f1_score_std']:.3f})")
    logger.info(f"  Perplexity: {domain_metrics['perplexity']:.2f}")
    return results

def save_baseline_results(results: Dict, save_path: str = "./results/baseline_performance.json") -> Path:
    """Save baseline results JSON (used by run_all)."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved to {p}")
    return p

