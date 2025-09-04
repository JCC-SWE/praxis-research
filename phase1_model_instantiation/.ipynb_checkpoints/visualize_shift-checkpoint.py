# visualize_shift.py
"""
Run-only script to generate distribution-shift visuals.
- Loads Qwen
- Pulls domain texts from your processed DAPT data (and optional second corpus)
- Saves PCA/UMAP/Hist/etc. plots to a timestamped folder
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# --- repo imports (make sure repo root is on sys.path) ---
_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../praxis-research
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "phase1_model_instantiation"))
    sys.path.insert(0, str(_REPO_ROOT))

from model_setup import load_qwen_model
from baseline_training import build_eval_data_auto
from model_validation import extract_embeddings, detect_distribution_shift


def _texts_from_processed(tokenizer, processed_path: Path, qa_jsonl: Path, max_domain_texts: int) -> list[str]:
    """Use the existing helper to harvest domain_texts from the test split."""
    qa_pairs, domain_texts = build_eval_data_auto(
        tokenizer=tokenizer,
        processed_data_path=str(processed_path),
        qa_jsonl_path=str(qa_jsonl),
        batch_size=8,
        max_domain_texts=max_domain_texts,
    )
    # We only need domain_texts here
    return domain_texts


def main():
    parser = argparse.ArgumentParser(description="Make distribution-shift visuals from embeddings.")
    parser.add_argument("--processed", default=str(_REPO_ROOT / "data" / "qwen_processed_data.pkl"),
                        help="Path to baseline processed .pkl")
    parser.add_argument("--processed_new", default=None,
                        help="Optional second processed .pkl for NEW data (if omitted, baseline is reused)")
    parser.add_argument("--qa", default=str(_REPO_ROOT / "data" / "qa_eval.jsonl"),
                        help="Held-out QA jsonl (only needed by loader; not used for visuals)")
    parser.add_argument("--max_texts", type=int, default=800,
                        help="Cap number of domain texts per corpus (speed/VRAM)")
    # 🔽 changed default here
    parser.add_argument("--outdir", 
                        default=str(_REPO_ROOT / "phase1_model_instantiation" / "results"),
                        help="Results directory (will auto-create timestamped subdir)")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix for figures")
    args = parser.parse_args()

    processed_path = Path(args.processed)
    qa_jsonl = Path(args.qa)
    out_root = Path(args.outdir)

    # Timestamped run folder inside results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"visuals_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Output directory: {run_dir}")

    print(f"🔧 Output directory: {run_dir}")

    # 1) Load model
    print("🧠 Loading model...")
    model, tokenizer = load_qwen_model()

    # 2) Build corpora (baseline + new)
    print("📥 Reading baseline domain texts…")
    base_texts = _texts_from_processed(tokenizer, processed_path, qa_jsonl, args.max_texts)

    if args.processed_new:
        print("📥 Reading NEW domain texts…")
        new_texts = _texts_from_processed(tokenizer, Path(args.processed_new), qa_jsonl, args.max_texts)
    else:
        print("⚠️  No --processed_new provided; using baseline texts as NEW (visuals still produced, shift likely not detected).")
        new_texts = base_texts[:]

    # 3) Extract embeddings
    print("🔎 Extracting embeddings…")
    base_emb = extract_embeddings(model, tokenizer, base_texts, max_length=512, batch_size=8)
    new_emb  = extract_embeddings(model, tokenizer, new_texts,  max_length=512, batch_size=8)

    # 4) Run detection + save visuals
    print("📊 Running distribution-shift detection + saving figures…")
    prefix = args.prefix or f"shift_{ts}"
    results = detect_distribution_shift(
        base_embeddings=base_emb,
        new_embeddings=new_emb,
        verbose=True,
        save_dir=str(run_dir),
        fig_prefix=prefix,
    )

    # 5) Summary
    print("\n=== SUMMARY ===")
    print(f"Shift detected: {results['shift_detected']} ({results['confidence']})")
    print(f"Evidence: {results['evidence_score']}/{results['max_evidence_score']}")
    if results.get("figures"):
        print("Saved figures:")
        for p in results["figures"]:
            print(f" - {p}")
    else:
        print("No figures saved.")

    # Also drop a tiny JSON for convenience
    import json
    with (run_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
