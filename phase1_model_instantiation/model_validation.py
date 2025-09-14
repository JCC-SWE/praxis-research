#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Evaluation: Base-token vs New-text embeddings
- Loads Qwen model + tokenizer
- BASE = token embedding matrix (optionally subsampled)
- NEW  = sequence embeddings from your 2025 corpus
- Diagnostics + distribution shift tests + visuals
- Artifacts saved under results/ModelEvaluation/<timestamp>/
"""

from __future__ import annotations

import os
import json
import time
import argparse
import pickle
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from model_setup import list_qwen_models
import numpy as np
import torch
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

# Matplotlib: non-interactive backend for headless saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


# -------------------------------------------------------------------
# Defaults / paths
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results" / "ModelEvaluation"


def _timestamp_dir(root: Path) -> Path:
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


# -------------------------------------------------------------------
# Model load
# -------------------------------------------------------------------
def load_qwen(model_name: str = "qwen-2.5-3b", local_path: str = "/workspace/praxis-research/base-model"):
    from transformers import AutoModel, AutoTokenizer
    from pathlib import Path
    
    logger.info(f"Loading model from storage: {model_name}")
    
    # Download model from blob storage
    model_path = get_qwen_model()
    
    if model_path is None:
        logger.error(f"Failed to download model {model_name}")
        raise Exception(f"Failed to download model {model_name} from storage")
    
    logger.info(f"Model downloaded to: {model_path}")
    
    # Load model and tokenizer from local path
    logger.info(f"Loading model/tokenizer from: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    logger.info(f"Successfully loaded model {model_name}")
    return mdl, tok


# -------------------------------------------------------------------
# Embedding utilities
# -------------------------------------------------------------------
def load_base_model_embeddings(
    model_or_path,
    sample_tokens: Optional[int] = None,
    random_state: int = 42,
    normalize_rows: bool = False,
) -> np.ndarray:
    """
    Load the model's token embedding matrix as BASE embeddings: (V, D).
    """
    if isinstance(model_or_path, str):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_or_path)
    else:
        m = model_or_path

    # Find embedding layer
    if hasattr(m, "embed_tokens"):
        emb = m.embed_tokens
    elif hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
        emb = m.model.embed_tokens
    else:
        emb = m.get_input_embeddings()

    if emb is None or not hasattr(emb, "weight"):
        raise RuntimeError("Could not locate input embedding layer (embed_tokens).")

    with torch.no_grad():
        W = emb.weight.detach().cpu().numpy()

    if sample_tokens is not None and sample_tokens < W.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(W.shape[0], sample_tokens, replace=False)
        W = W[idx]

    if normalize_rows:
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        W = W / norms

    logger.info(f"BASE token embeddings shape: {W.shape}")
    return W


def get_sequence_embeddings(
    model, tokenizer, texts: List[str], max_length: int = 512, batch_size: int = 8
) -> np.ndarray:
    """
    Mean-pool token embeddings over sequence length -> (N, D)
    """
    device = next(model.parameters()).device
    all_out: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)

            # Prefer embedding layer for stability
            if hasattr(model, "embed_tokens"):
                tok_emb = model.embed_tokens(inputs["input_ids"])
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                tok_emb = model.model.embed_tokens(inputs["input_ids"])
            else:
                outputs = model(**inputs, output_hidden_states=True)
                tok_emb = outputs.hidden_states[0]

            mask = inputs["attention_mask"].unsqueeze(-1)
            masked = tok_emb * mask
            denom = torch.clamp(mask.sum(dim=1), min=1)
            pooled = masked.sum(dim=1) / denom
            all_out.append(pooled.detach().cpu().numpy())

    arr = np.concatenate(all_out, axis=0)
    logger.info(f"NEW sequence embeddings shape: {arr.shape}")
    return arr


def diagnose_embeddings(embeddings: np.ndarray, name: str = "Embeddings") -> Dict[str, Any]:
    norms = np.linalg.norm(embeddings, axis=1)
    diag = {
        "shape": embeddings.shape,
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "extreme_count": int(np.sum((embeddings < -10) | (embeddings > 10))),
        "inf_count": int(np.sum(np.isinf(embeddings))),
        "nan_count": int(np.sum(np.isnan(embeddings))),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }
    logger.info(f"{name} diagnostics: {diag}")
    return diag


# -------------------------------------------------------------------
# NEW corpus loading
# -------------------------------------------------------------------
def decode_from_processed_pkl(pkl_path: Path, tokenizer, limit: int) -> List[str]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    dataset = data["dataset"]
    total = len(dataset)
    test_start = int(total * 0.9)
    test_ds = dataset.select(range(test_start, total))
    texts: List[str] = []
    n = min(len(test_ds), limit)
    for ex in test_ds.select(range(n)):
        ids = ex["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def load_new_texts(
    tokenizer,
    processed_pkl: Optional[str],
    texts_path: Optional[str],
    max_texts: int,
) -> List[str]:
    """
    Prefer processed .pkl (fast). Otherwise JSONL/TXT path.
    """
    if processed_pkl:
        p = Path(processed_pkl)
        if p.exists():
            logger.info(f"Decoding NEW texts from processed pickle: {p}")
            return decode_from_processed_pkl(p, tokenizer, max_texts)
        else:
            logger.warning(f"Processed pickle not found at {p}, falling back to texts_path.")

    if not texts_path:
        raise FileNotFoundError("Provide --new_processed_data_path or --new_texts_path.")

    p = Path(texts_path)
    if not p.exists():
        raise FileNotFoundError(f"NEW texts file not found: {p}")

    texts: List[str] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj.get("text"), str):
                        texts.append(obj["text"])
                    else:
                        vals = [str(v) for v in obj.values() if isinstance(v, str)]
                        if vals:
                            texts.append(" ".join(vals))
                except Exception:
                    continue
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)

    return texts[:max_texts]


# -------------------------------------------------------------------
# Comparison / shift metrics
# -------------------------------------------------------------------
def compare_base_vs_new_embeddings(
    base_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    base_sample_size: Optional[int] = 50000,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    BASE = token matrix (V, D) (optionally subsampled independently)
    NEW  = sequence embeddings (N, D)
    """
    logger.info(f"Base embeddings: {base_embeddings.shape} | New embeddings: {new_embeddings.shape}")

    # Independent cap for BASE only (NEW left intact)
    if base_sample_size is not None and base_embeddings.shape[0] > base_sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(base_embeddings.shape[0], base_sample_size, replace=False)
        base_sample = base_embeddings[idx]
    else:
        base_sample = base_embeddings.copy()

    logger.info(f"Using {len(base_sample)} BASE samples for comparison (independent cap)")

    # Align dims
    dim = min(base_sample.shape[1], new_embeddings.shape[1])
    base_sample = base_sample[:, :dim]
    new_clip = new_embeddings[:, :dim]

    # Clip to avoid outliers destabilizing histograms
    base_clipped = np.clip(base_sample, -10, 10)
    new_clipped = np.clip(new_clip, -10, 10)

    # Wasserstein per-dim
    wdists = []
    for d in range(dim):
        wd = wasserstein_distance(base_clipped[:, d], new_clipped[:, d])
        if np.isfinite(wd):
            wdists.append(wd)
    wdists = np.array(wdists, dtype=float)

    # Jensen-Shannon per-dim (hist-based)
    js = []
    for d in range(dim):
        bv = base_clipped[:, d]
        nv = new_clipped[:, d]
        bins = np.linspace(min(bv.min(), nv.min()), max(bv.max(), nv.max()), 51)
        h1, _ = np.histogram(bv, bins=bins, density=True)
        h2, _ = np.histogram(nv, bins=bins, density=True)
        h1 = h1 / (np.sum(h1) + 1e-15) + 1e-15
        h2 = h2 / (np.sum(h2) + 1e-15) + 1e-15
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        jsd = jensenshannon(h1, h2)
        js.append(0.0 if not np.isfinite(jsd) else float(jsd))
    js = np.array(js, dtype=float)

    results = {
        "avg_wasserstein": float(np.mean(wdists)) if len(wdists) else 0.0,
        "max_wasserstein": float(np.max(wdists)) if len(wdists) else 0.0,
        "std_wasserstein": float(np.std(wdists)) if len(wdists) else 0.0,
        "avg_js": float(np.mean(js)) if len(js) else 0.0,
        "max_js": float(np.max(js)) if len(js) else 0.0,
        "std_js": float(np.std(js)) if len(js) else 0.0,
        "base_mean": float(np.mean(base_clipped)),
        "new_mean": float(np.mean(new_clipped)),
        "base_std": float(np.std(base_clipped)),
        "new_std": float(np.std(new_clipped)),
        "base_extreme_count": int(np.sum((base_sample < -10) | (base_sample > 10))),
        "new_extreme_count": int(np.sum((new_clip < -10) | (new_clip > 10))),
        "dims": int(dim),
    }
    return results, base_sample


def detect_distribution_shift(
    base_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    significance_level: float = 0.05,
    base_cap: Optional[int] = 50000,
    new_cap: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Multi-test decision with independent caps for BASE/NEW.
    """
    # Independent sampling
    rng = np.random.default_rng(42)

    if base_cap is not None and len(base_embeddings) > base_cap:
        base_idx = rng.choice(len(base_embeddings), base_cap, replace=False)
        base_sample = base_embeddings[base_idx]
    else:
        base_sample = base_embeddings.copy()

    if new_cap is not None and len(new_embeddings) > new_cap:
        new_idx = rng.choice(len(new_embeddings), new_cap, replace=False)
        new_sample = new_embeddings[new_idx]
    else:
        new_sample = new_embeddings.copy()

    # Align dims
    dim = min(base_sample.shape[1], new_sample.shape[1])
    base_sample = base_sample[:, :dim]
    new_sample = new_sample[:, :dim]

    if verbose:
        print(f"Comparing {len(base_sample)} base vs {len(new_sample)} new samples")
        print(f"Embedding dimension: {dim}")

    results: Dict[str, Any] = {
        "shift_detected": False,
        "confidence": "None",
        "evidence_score": 0,
        "max_evidence_score": 8,
        "tests": {},
    }

    # Test 1: extreme values
    b_ext = float(np.mean(np.abs(base_sample) > 10))
    n_ext = float(np.mean(np.abs(new_sample) > 10))
    diff = abs(n_ext - b_ext)
    results["tests"]["extreme_values"] = {
        "base_extreme_pct": b_ext,
        "new_extreme_pct": n_ext,
        "difference": diff,
        "significant": diff > 0.001,  # 0.1%
    }
    if diff > 0.001:
        results["evidence_score"] += 2

    # Test 2: norms (Welch t)
    base_clipped = np.clip(base_sample, -50, 50)
    new_clipped = np.clip(new_sample, -50, 50)
    bnorm = np.linalg.norm(base_clipped, axis=1)
    nnorm = np.linalg.norm(new_clipped, axis=1)
    try:
        _, pval = stats.ttest_ind(bnorm, nnorm, equal_var=False)
        ok = np.isfinite(pval)
    except Exception:
        pval, ok = 1.0, False
    ratio = float(np.mean(nnorm) / (np.mean(bnorm) if np.mean(bnorm) > 0 else 1.0))
    results["tests"]["norm_shift"] = {
        "norm_ratio": ratio,
        "t_pvalue": float(pval),
        "significant": bool(ok and pval < significance_level),
    }
    if ok and pval < significance_level:
        results["evidence_score"] += 2
    elif ratio > 2.0 or ratio < 0.5:
        results["evidence_score"] += 1

    # Test 3: Wasserstein (first 20 dims)
    dtest = min(20, dim)
    wd = []
    for d in range(dtest):
        w = wasserstein_distance(base_clipped[:, d], new_clipped[:, d])
        if np.isfinite(w):
            wd.append(w)
    avg_w = float(np.mean(wd)) if wd else 0.0
    results["tests"]["wasserstein"] = {
        "average_distance": avg_w,
        "max_distance": float(np.max(wd)) if wd else 0.0,
        "dimensions_tested": len(wd),
        "significant": avg_w > 0.5,
    }
    if avg_w > 1.0:
        results["evidence_score"] += 2
    elif avg_w > 0.5:
        results["evidence_score"] += 1

    # Test 4: PCA + KS
    try:
        combined = np.vstack([base_clipped, new_clipped])
        pca = PCA(n_components=min(5, dim))
        pca.fit(combined)
        bpc = pca.transform(base_clipped)
        npc = pca.transform(new_clipped)
        ks = []
        for i in range(min(3, bpc.shape[1])):
            _, pv = stats.ks_2samp(bpc[:, i], npc[:, i])
            if np.isfinite(pv):
                ks.append(pv)
        sig = int(np.sum(np.array(ks) < significance_level)) if ks else 0
        results["tests"]["pca_shift"] = {
            "significant_components": sig,
            "total_components": len(ks),
            "explained_variance_ratio": pca.explained_variance_ratio_[: min(3, len(pca.explained_variance_ratio_))].tolist(),
        }
        if sig >= 2:
            results["evidence_score"] += 2
        elif sig >= 1:
            results["evidence_score"] += 1
    except Exception as e:
        results["tests"]["pca_shift"] = {"error": str(e)}

    # Final decision
    er = results["evidence_score"] / results["max_evidence_score"]
    if er >= 0.75:
        results["shift_detected"] = True
        results["confidence"] = "High"
    elif er >= 0.5:
        results["shift_detected"] = True
        results["confidence"] = "Moderate"
    elif er >= 0.25:
        results["shift_detected"] = True
        results["confidence"] = "Low"
    else:
        results["shift_detected"] = False
        results["confidence"] = "None"

    return results


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------
def visualize_base_vs_new(
    base_sample: np.ndarray,
    new_embeddings: np.ndarray,
    out_dir: Path,
    prefix: str = "base_vs_new",
) -> List[str]:
    """
    Saves:
      - PCA scatter
      - UMAP scatter (if available)
      - PC1 histogram (base outline, new filled)
      - Norm histogram (base outline, new filled)
      - Per-dim means (first 100)
      - Per-dim stds (first 100)
    """
    paths: List[str] = []
    combined = np.vstack([base_sample, new_embeddings])
    labels = np.array(["Base"] * len(base_sample) + ["2025 AI"] * len(new_embeddings))
    palette = {"Base": "tab:blue", "2025 AI": "tab:red"}

    # PCA scatter
    pca = PCA(n_components=2)
    p2 = pca.fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["Base", "2025 AI"]:
        m = labels == name
        ax.scatter(p2[m, 0], p2[m, 1], s=10, alpha=0.6, label=name, c=palette[name])
    ax.set_title(f"PCA (2D) — total explained {pca.explained_variance_ratio_.sum():.3f}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
    ax.grid(True, alpha=0.3); ax.legend()
    p = out_dir / f"{prefix}_pca_scatter.png"
    fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    paths.append(str(p))

    # UMAP scatter (optional)
    try:
        import umap  # type: ignore
        u2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(combined)
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in ["Base", "2025 AI"]:
            m = labels == name
            ax.scatter(u2[m, 0], u2[m, 1], s=10, alpha=0.6, label=name, c=palette[name])
        ax.set_title("UMAP (2D)"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3); ax.legend()
        p = out_dir / f"{prefix}_umap_scatter.png"
        fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
        paths.append(str(p))
    except Exception as e:
        logger.info(f"UMAP not available: {e}")

    # PC1 hist overlay
    bp = pca.transform(base_sample)[:, 0]
    np1 = pca.transform(new_embeddings)[:, 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(min(bp.min(), np1.min()), max(bp.max(), np1.max()), 60)
    ax.hist(bp, bins=bins, density=True, histtype="step", linewidth=2, label="Base", color=palette["Base"])
    ax.hist(np1, bins=bins, density=True, histtype="stepfilled", alpha=0.35, label="2025 AI", color=palette["2025 AI"])
    ax.set_title("PC1 Distribution"); ax.set_xlabel("PC1 value"); ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3); ax.legend()
    p = out_dir / f"{prefix}_pc1_hist.png"
    fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    paths.append(str(p))

    # Norms hist overlay
    bnorm = np.linalg.norm(base_sample, axis=1)
    nnorm = np.linalg.norm(new_embeddings, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(min(bnorm.min(), nnorm.min()), max(bnorm.max(), nnorm.max()), 60)
    ax.hist(bnorm, bins=bins, density=True, histtype="step", linewidth=2, label="Base", color=palette["Base"])
    ax.hist(nnorm, bins=bins, density=True, histtype="stepfilled", alpha=0.35, label="2025 AI", color=palette["2025 AI"])
    ax.set_title("Embedding Norms"); ax.set_xlabel("L2 norm"); ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3); ax.legend()
    p = out_dir / f"{prefix}_norms_hist.png"
    fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    paths.append(str(p))

    # Per-dim means/stds (first 100)
    dims = min(100, base_sample.shape[1])
    bmeans = np.mean(base_sample[:, :dims], axis=0); nmeans = np.mean(new_embeddings[:, :dims], axis=0)
    bstds  = np.std(base_sample[:, :dims], axis=0);  nstds  = np.std(new_embeddings[:, :dims], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bmeans, label="Base mean", alpha=0.9, color=palette["Base"])
    ax.plot(nmeans, label="2025 AI mean",  alpha=0.9, color=palette["2025 AI"])
    ax.set_title(f"Per-dimension Means (first {dims} dims)"); ax.set_xlabel("Dimension"); ax.set_ylabel("Mean")
    ax.grid(True, alpha=0.3); ax.legend()
    p = out_dir / f"{prefix}_means_first{dims}.png"
    fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bstds, label="Base std", alpha=0.9, color=palette["Base"])
    ax.plot(nstds, label="2025 AI std",  alpha=0.9, color=palette["2025 AI"])
    ax.set_title(f"Per-dimension STDs (first {dims} dims)"); ax.set_xlabel("Dimension"); ax.set_ylabel("Std")
    ax.grid(True, alpha=0.3); ax.legend()
    p = out_dir / f"{prefix}_stds_first{dims}.png"
    fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    paths.append(str(p))

    return paths


# -------------------------------------------------------------------
# CLI runner
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model evaluation: base token embeddings vs 2025 corpus embeddings")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--new_processed_data_path",
                        default=str(REPO_ROOT.parent / "data" / "qwen_processed_data.pkl"),
                        help="Processed 2025 corpus (.pkl). Preferred.")
    parser.add_argument("--new_texts_path", default=None, help="Fallback JSONL/TXT if no .pkl.")
    parser.add_argument("--max_texts", type=int, default=512, help="Max decoded NEW texts.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_sample_size", type=int, default=50000, help="Cap for BASE token rows; -1 for all.")
    parser.add_argument("--normalize_base_rows", action="store_true", help="Row-normalize BASE token vectors.")
    parser.add_argument("--save_dir", default=None, help="Optional custom output dir.")
    args = parser.parse_args()

    out_dir = Path(args.save_dir) if args.save_dir else _timestamp_dir(RESULTS_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts => {out_dir}")

    # 1) Load model/tokenizer
    model, tokenizer = load_qwen(args.model_name)

    # 2) BASE = token embedding matrix (independent cap)
    cap = None if args.base_sample_size == -1 else args.base_sample_size
    base_emb = load_base_model_embeddings(
        model, sample_tokens=cap, normalize_rows=args.normalize_base_rows
    )
    base_diag = diagnose_embeddings(base_emb, "Base token embeddings")

    # 3) NEW = 2025 texts -> sequence embeddings
    new_texts = load_new_texts(
        tokenizer,
        processed_pkl=args.new_processed_data_path,
        texts_path=args.new_texts_path,
        max_texts=args.max_texts,
    )
    logger.info(f"Loaded {len(new_texts)} NEW texts")
    new_emb = get_sequence_embeddings(model, tokenizer, new_texts, args.max_length, args.batch_size)
    new_diag = diagnose_embeddings(new_emb, "2025 sequence embeddings")

    # 4) Compare + shift detection
    cmp_results, base_sample = compare_base_vs_new_embeddings(base_emb, new_emb, base_sample_size=cap)
    shift = detect_distribution_shift(
        base_emb, new_emb, significance_level=0.05, base_cap=cap, new_cap=None, verbose=True
    )

    # 5) Visuals
    figs = visualize_base_vs_new(base_sample, new_emb, out_dir, prefix="eval")

    # 6) Save results
    results = {
        "experiment": {
            "model": args.model_name,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "out_dir": str(out_dir),
        },
        "diagnostics": {
            "base": base_diag,
            "new": new_diag,
        },
        "comparison": cmp_results,
        "shift": shift,
        "figures": figs,
    }
    _save_json(results, out_dir / "results.json")

    # Console summary
    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Artifacts saved to: {out_dir}")
    print(f"Figures:\n- " + "\n- ".join(figs))
    print("\nKey metrics:")
    print(f"- Avg Wasserstein: {cmp_results['avg_wasserstein']:.4f}")
    print(f"- Avg JS:          {cmp_results['avg_js']:.4f}")
    print(f"- Norm ratio:      {shift['tests']['norm_shift']['norm_ratio']:.3f}")
    print(f"- Shift detected:  {shift['shift_detected']} (confidence: {shift['confidence']})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
