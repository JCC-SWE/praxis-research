"""
run_all.py - Complete baseline evaluation pipeline
Orchestrates model loading, baseline evaluation, distribution shift detection
Saves results in report-ready format with timestamps
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (optional; kept for future plots)
from typing import Dict, List, Any

# Import our modules
from model_setup import load_qwen_model, save_model_checkpoint, get_model_info
from model_validation import extract_embeddings, detect_distribution_shift
from baseline_training import establish_baseline_performance, save_baseline_results, build_eval_data_auto


import sys
from pathlib import Path



class PraxisExperimentRunner:
    """Main experiment runner for Neural Network Surgery praxis research"""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped experiment directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize results storage
        self.results = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "start_time": datetime.now().isoformat(),
                "experiment_id": f"baseline_eval_{self.timestamp}",
                "phase": "baseline_evaluation",
            },
            "model_info": {},
            "baseline_performance": {},
            "distribution_shift": {},
            "embeddings_analysis": {},
            "execution_log": [],
            "metrics": [],  # <- for log_metric
        }

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.experiment_dir / f"experiment_log_{self.timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def log_step(self, step_name: str, details: Dict[str, Any] = None):
        """Log experiment step with timestamp"""
        step_info = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        self.results["execution_log"].append(step_info)
        self.logger.info(f"STEP: {step_name}")

    def log_metric(self, category: str, metric: str, value: Any, unit: str = ""):
        """Record a metric for later CSV/reporting."""
        self.results["metrics"].append(
            {
                "category": category,
                "metric": metric,
                "value": value,
                "unit": unit,
                "timestamp": self.timestamp,
            }
        )

    def load_and_analyze_model(self, model_checkpoint_path: str = None):
        """Load model and capture comprehensive information"""
        self.log_step("Loading Qwen 2.5-3B Model")

        start_time = time.time()

        if model_checkpoint_path and Path(model_checkpoint_path).exists():
            self.logger.info(f"Loading from checkpoint: {model_checkpoint_path}")
            from model_setup import load_model_checkpoint

            model, tokenizer = load_model_checkpoint(model_checkpoint_path)
        else:
            self.logger.info("Loading from HuggingFace Hub")
            model, tokenizer = load_qwen_model()

        load_time = time.time() - start_time

        # Get comprehensive model info
        model_info = get_model_info(model)
        device_info = str(next(model.parameters()).device)

        self.results["model_info"] = {
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "load_time_seconds": load_time,
            "device": device_info,
            "parameters": model_info,
            "checkpoint_used": model_checkpoint_path is not None,
            "checkpoint_path": str(model_checkpoint_path) if model_checkpoint_path else None,
        }

        # Log metrics to CSV
        self.log_metric("model", "total_parameters", model_info["total_parameters"], "count")
        self.log_metric("model", "trainable_parameters", model_info["trainable_parameters"], "count")
        self.log_metric("model", "model_size_gb", model_info["size_gb"], "GB")
        self.log_metric("model", "load_time", load_time, "seconds")

        self.log_step(
            "Model Loaded Successfully",
            {
                "load_time": f"{load_time:.2f}s",
                "total_params": model_info["total_parameters"],
                "device": device_info,
            },
        )

        return model, tokenizer

    def run_baseline_evaluation(
        self, model, tokenizer, qa_pairs: List[Dict], domain_texts: List[str]
    ):
        """Run comprehensive baseline evaluation"""
        self.log_step("Running Baseline Performance Evaluation")

        start_time = time.time()

        # Run baseline evaluation
        baseline_results = establish_baseline_performance(model, tokenizer, qa_pairs, domain_texts)

        evaluation_time = time.time() - start_time

        # Add timing and metadata
        baseline_results["evaluation_metadata"] = {
            "evaluation_time_seconds": evaluation_time,
            "timestamp": datetime.now().isoformat(),
            "num_qa_pairs": len(qa_pairs),
            "num_domain_texts": len(domain_texts),
        }

        self.results["baseline_performance"] = baseline_results

        # Log metrics to CSV
        qa_metrics = baseline_results["qa_metrics"]
        domain_metrics = baseline_results["domain_metrics"]

        self.log_metric("baseline", "exact_match", qa_metrics["exact_match"], "ratio")
        self.log_metric("baseline", "exact_match_std", qa_metrics["exact_match_std"], "ratio")
        self.log_metric("baseline", "f1_score", qa_metrics["f1_score"], "ratio")
        self.log_metric("baseline", "f1_score_std", qa_metrics["f1_score_std"], "ratio")
        self.log_metric("baseline", "perplexity", domain_metrics["perplexity"], "score")
        self.log_metric("baseline", "log_perplexity", domain_metrics["log_perplexity"], "score")
        self.log_metric("baseline", "evaluation_time", evaluation_time, "seconds")
        self.log_metric("baseline", "num_qa_pairs", len(qa_pairs), "count")
        self.log_metric("baseline", "num_domain_texts", len(domain_texts), "count")

        self.log_step(
            "Baseline Evaluation Complete",
            {
                "evaluation_time": f"{evaluation_time:.2f}s",
                "exact_match": f"{baseline_results['qa_metrics']['exact_match']:.3f}",
                "f1_score": f"{baseline_results['qa_metrics']['f1_score']:.3f}",
                "perplexity": f"{baseline_results['domain_metrics']['perplexity']:.2f}",
            },
        )

        return baseline_results

    def run_distribution_shift_analysis(self, model, tokenizer, baseline_texts: List[str], new_texts: List[str]):
        """Run comprehensive distribution shift detection"""
        self.log_step("Extracting Embeddings for Distribution Shift Analysis")
        
        # Extract embeddings
        start_time = time.time()
        baseline_embeddings = extract_embeddings(model, tokenizer, baseline_texts)
        new_embeddings = extract_embeddings(model, tokenizer, new_texts)
        embedding_time = time.time() - start_time
        
        self.log_step("Running Distribution Shift Detection Tests")
        
        # >>> NEW: figure output dir per experiment
        fig_dir = self.experiment_dir / "figures"
        fig_prefix = f"shift_{self.timestamp}"
        
        # Run distribution shift detection (now saves figures)
        start_time = time.time()
        shift_results = detect_distribution_shift(
            baseline_embeddings,
            new_embeddings,
            verbose=True,
            save_dir=str(fig_dir),
            fig_prefix=fig_prefix,
        )
        detection_time = time.time() - start_time
        
        # Analyze embeddings statistics (unchanged)
        embeddings_stats = self.analyze_embeddings_statistics(baseline_embeddings, new_embeddings)
        
        # >>> include figure paths in results
        self.results["distribution_shift"] = {
            **shift_results,
            "detection_metadata": {
                "embedding_extraction_time": embedding_time,
                "detection_time": detection_time,
                "timestamp": datetime.now().isoformat(),
                "baseline_texts_count": len(baseline_texts),
                "new_texts_count": len(new_texts),
                "figures_dir": str(fig_dir),
            },
        }
        
        self.results["embeddings_analysis"] = embeddings_stats
        
        # Log metrics (existing) ...
        # ...
        
        # >>> NEW: log figure paths & counts for CSV convenience
        fig_paths = shift_results.get("figures", [])
        self.log_metric("distribution_shift", "figures_count", len(fig_paths), "count")
        if fig_paths:
            self.log_metric("distribution_shift", "figures_dir", str(fig_dir), "path")
            for p in fig_paths:
                self.log_metric("figure", "path", str(p), "path")
        
        self.log_step(
            "Distribution Shift Analysis Complete",
            {
                "shift_detected": shift_results["shift_detected"],
                "confidence": shift_results["confidence"],
                "evidence_score": f"{shift_results['evidence_score']}/{shift_results['max_evidence_score']}",
                "figures_saved": len(fig_paths),
                "figures_dir": str(fig_dir),
            },
        )
        
        return shift_results, embeddings_stats


    def analyze_embeddings_statistics(self, baseline_embeddings, new_embeddings):
        """Analyze embedding statistics for academic reporting"""

        def embedding_stats(embeddings, name):
            return {
                "name": name,
                "shape": embeddings.shape,
                "mean": float(np.mean(embeddings)),
                "std": float(np.std(embeddings)),
                "min": float(np.min(embeddings)),
                "max": float(np.max(embeddings)),
                "norm_mean": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "norm_std": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "extreme_values_count": int(np.sum(np.abs(embeddings) > 10)),
                "extreme_values_percentage": float(
                    np.sum(np.abs(embeddings) > 10) / embeddings.size * 100
                ),
                "nan_count": int(np.sum(np.isnan(embeddings))),
                "inf_count": int(np.sum(np.isinf(embeddings))),
            }

        return {
            "baseline_embeddings": embedding_stats(baseline_embeddings, "baseline"),
            "new_embeddings": embedding_stats(new_embeddings, "new_data"),
            "comparison": {
                "dimensionality_match": baseline_embeddings.shape[1] == new_embeddings.shape[1],
                "baseline_samples": baseline_embeddings.shape[0],
                "new_samples": new_embeddings.shape[0],
            },
        }

    def save_comprehensive_results(self):
        """Save all results in multiple formats for academic reporting"""

        # Add completion timestamp
        self.results["experiment_info"]["end_time"] = datetime.now().isoformat()
        self.results["experiment_info"]["total_duration"] = str(
            datetime.now() - datetime.fromisoformat(self.results["experiment_info"]["start_time"])
        )

        # Save main results JSON
        results_file = self.experiment_dir / "complete_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary report
        self.generate_summary_report()

        # Save metrics CSV for analysis
        self.generate_metrics_csv()

        self.log_step("Results Saved", {"output_directory": str(self.experiment_dir)})

        return self.experiment_dir

    def generate_summary_report(self):
        """Generate human-readable summary report"""
        report_file = self.experiment_dir / "experiment_summary.md"
    
        with open(report_file, "w") as f:
            f.write("# Neural Network Surgery Praxis - Baseline Evaluation\n\n")
            f.write(f"**Experiment ID:** {self.results['experiment_info']['experiment_id']}\n")
            f.write(f"**Timestamp:** {self.results['experiment_info']['timestamp']}\n")
            f.write(f"**Duration:** {self.results['experiment_info']['total_duration']}\n\n")
    
            f.write(f"- **Model:** {self.results['model_info']['model_name']}\n")
            f.write(f"- **Parameters:** {self.results['model_info']['parameters']['total_parameters']:,}\n")
            f.write(f"- **Device:** {self.results['model_info']['device']}\n")
            f.write(f"- **Load Time:** {self.results['model_info']['load_time_seconds']:.2f}s\n\n")
    
            if self.results.get("baseline_performance"):
                bp = self.results["baseline_performance"]
                f.write("## Baseline Performance Metrics\n")
                f.write(f"- **Exact Match:** {bp['qa_metrics']['exact_match']:.3f} "
                        f"(±{bp['qa_metrics']['exact_match_std']:.3f})\n")
                f.write(f"- **F1 Score:** {bp['qa_metrics']['f1_score']:.3f} "
                        f"(±{bp['qa_metrics']['f1_score_std']:.3f})\n")
                f.write(f"- **Perplexity:** {bp['domain_metrics']['perplexity']:.2f}\n")
                f.write(f"- **Questions Evaluated:** {bp['qa_metrics']['num_questions']}\n")
                f.write(f"- **Domain Texts:** {bp['domain_metrics']['num_texts']}\n\n")
    
            if self.results.get("distribution_shift"):
                ds = self.results["distribution_shift"]
                f.write("## Distribution Shift Detection\n")
                f.write(f"- **Shift Detected:** {'✅ YES' if ds['shift_detected'] else '❌ NO'}\n")
                f.write(f"- **Confidence Level:** {ds['confidence']}\n")
                f.write(f"- **Evidence Score:** {ds['evidence_score']}/{ds['max_evidence_score']}\n\n")
    
                f.write("### Statistical Test Results\n")
                for test_name, test_result in ds.get("tests", {}).items():
                    label = "✅ Significant" if test_result.get("significant", False) else "❌ Not Significant"
                    f.write(f"- **{test_name.replace('_', ' ').title()}:** {label}\n")
                f.write("\n")
    
                # NEW: Figure paths
                figs = ds.get("figures", [])
                if figs:
                    fig_dir = ds.get("detection_metadata", {}).get("figures_dir", "")
                    f.write("### Figures\n")
                    if fig_dir:
                        f.write(f"- **Directory:** {fig_dir}\n")
                    for p in figs:
                        f.write(f"- {p}\n")
                    f.write("\n")
    
            f.write("## Execution Log\n")
            for step in self.results.get("execution_log", []):
                f.write(f"- **{step['timestamp']}:** {step['step']}\n")


    def generate_metrics_csv(self):
        """Generate CSV of key metrics for analysis"""
        # Prefer metrics collected via log_metric
        metrics_data = list(self.results.get("metrics", []))

        # Also ensure a few core metrics exist
        if "baseline_performance" in self.results and self.results["baseline_performance"]:
            bp = self.results["baseline_performance"]
            # Avoid duplicates: only add if not already logged
            already = {(m["category"], m["metric"]) for m in metrics_data}
            additions = [
                ("baseline", "exact_match", bp["qa_metrics"]["exact_match"], "ratio"),
                ("baseline", "f1_score", bp["qa_metrics"]["f1_score"], "ratio"),
                ("baseline", "perplexity", bp["domain_metrics"]["perplexity"], "score"),
            ]
            for cat, met, val, unit in additions:
                if (cat, met) not in already:
                    metrics_data.append(
                        {
                            "category": cat,
                            "metric": met,
                            "value": val,
                            "unit": unit,
                            "timestamp": self.timestamp,
                        }
                    )

        df = pd.DataFrame(metrics_data)
        csv_file = self.experiment_dir / "metrics.csv"
        df.to_csv(csv_file, index=False)


def run_complete_baseline_experiment(
    qa_pairs: List[Dict],
    domain_texts: List[str],
    baseline_texts: List[str],
    new_texts: List[str],
    model_checkpoint_path: str = None,
):
    """
    Run complete baseline evaluation experiment
    """
    runner = PraxisExperimentRunner()
    try:
        model, tokenizer = runner.load_and_analyze_model(model_checkpoint_path)

        # Baseline eval
        runner.run_baseline_evaluation(model, tokenizer, qa_pairs, domain_texts)

        # Distribution shift
        runner.run_distribution_shift_analysis(model, tokenizer, baseline_texts, new_texts)

        # Persist
        results_dir = runner.save_comprehensive_results()

        print(f"\n{'='*60}")
        print(f"BASELINE MODEL EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {results_dir}")
        print(f"Experiment ID: {runner.results['experiment_info']['experiment_id']}")
        print(f"Duration: {runner.results['experiment_info']['total_duration']}")
        print(f"{'='*60}")

        return results_dir
    except Exception as e:
        runner.logger.error(f"Baseline evaluation failed: {str(e)}")
        raise


def main():
    # Orchestrate the simple baseline flow using DAPT texts + held-out QA
    runner = PraxisExperimentRunner()
    model, tokenizer = runner.load_and_analyze_model(model_checkpoint_path=None)

    # Build absolute paths for data
    REPO_ROOT = Path(__file__).resolve().parents[1]
    processed_abs = (REPO_ROOT / "data" / "qwen_processed_data.pkl").as_posix()
    qa_abs        = (REPO_ROOT / "data" / "qa_eval.jsonl").as_posix()

    # Build eval inputs: QA from held-out JSONL (if needed), domain texts from DAPT test set
    qa_pairs, domain_texts = build_eval_data_auto(
        tokenizer=tokenizer,
        processed_data_path=processed_abs,
        qa_jsonl_path=qa_abs,
        batch_size=8,
        max_domain_texts=512,  # optional cap for speed
    )

    runner.run_baseline_evaluation(model, tokenizer, qa_pairs, domain_texts)

    # (Optional) distribution shift later if you supply new_texts
    runner.save_comprehensive_results()
if __name__ == "__main__":
    main()

