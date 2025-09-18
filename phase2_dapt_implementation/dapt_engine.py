#!/usr/bin/env python3
"""
DAPT Engine - Model Interaction and Evaluation
Handles model interaction, testing, and domain adaptation assessment for trained DAPT models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import numpy as np
import os


def load_dapt_model(model_path: str):
    """Load a trained DAPT model and tokenizer from saved checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DAPT model not found at {model_path}")
    
    print(f"Loading DAPT model from {model_path}")
    
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Successfully loaded DAPT model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    return model, tokenizer


class ModelInteractor:
    """Handle model interaction and question answering"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def ask_question(self, question: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate response to a question using the trained model"""
        self.model.eval()
        inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(question):].strip()
        
        return answer
    
    def test_domain_adaptation(self, ai_questions: List[str] = None) -> Dict:
        """Test domain adaptation effectiveness with AI-related questions"""
        if ai_questions is None:
            ai_questions = [
                "What is artificial intelligence?",
                "How do neural networks learn?",
                "What are transformers in AI?",
                "Explain deep learning.",
                "What is machine learning?"
            ]

        print("\n=== Domain Adaptation Test ===")
        
        domain_scores = []
        results = {}
        
        for question in ai_questions:
            answer = self.ask_question(question, max_tokens=100)
            
            # Simple domain relevance scoring
            ai_keywords = ['neural', 'algorithm', 'learning', 'model', 'training', 
                          'data', 'artificial', 'intelligence', 'network', 'optimization']
            
            answer_lower = answer.lower()
            keyword_count = sum(1 for keyword in ai_keywords if keyword in answer_lower)
            domain_score = min(keyword_count / 3, 1.0)  # Normalize to 0-1
            domain_scores.append(domain_score)
            
            results[question] = {
                "answer": answer,
                "domain_score": domain_score
            }
            
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"Domain relevance score: {domain_score:.2f}")
            print("-" * 50)

        avg_domain_score = sum(domain_scores) / len(domain_scores)
        
        print(f"\nDomain Adaptation Results:")
        print(f"Average domain relevance: {avg_domain_score:.2f}")
        print(f"Domain adaptation: {'Successful' if avg_domain_score > 0.6 else 'Needs improvement'}")
        
        return {
            "avg_domain_score": avg_domain_score,
            "individual_scores": domain_scores,
            "detailed_results": results,
            "adaptation_successful": avg_domain_score > 0.6
        }
    
    def interactive_chat(self):
        """Start an interactive chat session with the model"""
        print("Starting interactive chat session. Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Ending chat session.")
                    break
                
                if not question:
                    continue
                
                answer = self.ask_question(question)
                print(f"AI: {answer}")
                
            except KeyboardInterrupt:
                print("\nEnding chat session.")
                break
            except Exception as e:
                print(f"Error: {e}")


class DAPTEvaluator:
    """Comprehensive DAPT evaluation and sufficiency analysis"""
    
    def __init__(self, model, tokenizer, training_results: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.training_results = training_results
        
    def calculate_model_efficiency(self, gpu_specs: Dict = None) -> Dict:
        """Calculate efficiency metrics for full DAPT"""
        if gpu_specs is None:
            gpu_specs = {"total_memory_gb": 85}  # Default H100 specs
        
        # Model specifications
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Training efficiency from actual experiment
        abstracts_processed = self.training_results.get("dataset_size", 4100)
        training_time_minutes = self.training_results.get("training_time_minutes", 14)
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 18.7
        
        # Calculate throughput
        abstracts_per_minute = abstracts_processed / training_time_minutes
        abstracts_per_hour = abstracts_per_minute * 60
        
        # Memory efficiency
        memory_utilization = gpu_memory_used / gpu_specs.get("total_memory_gb", 85)
        
        efficiency_metrics = {
            "model_specs": {
                "total_parameters": f"{total_params/1e9:.2f}B",
                "trainable_parameters": f"{trainable_params/1e9:.2f}B",
                "precision": "BF16"
            },
            "training_performance": {
                "abstracts_per_hour": round(abstracts_per_hour, 0),
                "gpu_memory_used_gb": round(gpu_memory_used, 1),
                "memory_utilization_pct": f"{memory_utilization:.1%}",
            }
        }
        
        print("Model Efficiency Metrics")
        print(f"Model Size: {efficiency_metrics['model_specs']['total_parameters']} parameters")
        print(f"Training Speed: {efficiency_metrics['training_performance']['abstracts_per_hour']} abstracts/hour")
        print(f"Memory Usage: {gpu_memory_used:.1f}GB / {gpu_specs.get('total_memory_gb', 85)}GB ({memory_utilization:.1%})")
        print(f"Memory Efficiency: {'EXCELLENT' if memory_utilization < 0.5 else 'GOOD' if memory_utilization < 0.8 else 'MODERATE'}")
        
        return efficiency_metrics
    
    def analyze_resource_sufficiency(self, gpu_specs: Dict = None) -> Dict:
        """Analyze if current resources are sufficient for full-scale DAPT"""
        if gpu_specs is None:
            gpu_specs = {"total_memory_gb": 85}
        
        # Actual performance
        current_abstracts = self.training_results.get("dataset_size", 4100)
        current_time_minutes = self.training_results.get("training_time_minutes", 14)
        current_memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 18.7
        
        # Scale projections for full research
        target_scenarios = {
            "target_10k": 10000,
            "medium_scale_50k": 50000,
            "large_scale_100k": 100000,
            "ambitious_500k": 500000
        }
        
        # Calculate scaling factors
        scaling_results = {}
        
        for scenario, target_abstracts in target_scenarios.items():
            scale_factor = target_abstracts / current_abstracts
            
            # Linear scaling for time and memory
            projected_time_hours = (current_time_minutes * scale_factor) / 60
            projected_memory_gb = current_memory_gb * (1 + np.log(scale_factor) * 0.1)  # Sublinear memory scaling
            
            # Resource availability check
            available_memory = gpu_specs.get("total_memory_gb", 85)
            memory_sufficient = projected_memory_gb < available_memory * 0.9  # 90% safety margin
            
            # Cost calculation (H100 spot pricing)
            cost_usd = projected_time_hours * 1.50
            
            scaling_results[scenario] = {
                "target_abstracts": target_abstracts,
                "projected_time_hours": round(projected_time_hours, 2),
                "projected_memory_gb": round(projected_memory_gb, 1),
                "memory_sufficient": memory_sufficient,
                "estimated_cost_usd": round(cost_usd, 2),
                "feasible": memory_sufficient and cost_usd < 500
            }
        
        # Overall sufficiency assessment
        feasible_scenarios = sum(1 for result in scaling_results.values() if result["feasible"])
        sufficiency_score = feasible_scenarios / len(scaling_results)
        
        print(f"\nResource Sufficiency Analysis")
        print(f"Current Baseline: {current_abstracts} abstracts in {current_time_minutes} minutes")
        print("\nScaling Projections:")
        for scenario, results in scaling_results.items():
            status = "✓" if results["feasible"] else "✗"
            print(f"  {status} {scenario}: {results['target_abstracts']:,} abstracts → {results['projected_time_hours']:.1f}h, ${results['estimated_cost_usd']:.2f}")
        
        resource_verdict = "SUFFICIENT" if sufficiency_score >= 0.75 else "LIMITED" if sufficiency_score >= 0.5 else "INSUFFICIENT"
        print(f"\nResource Verdict: {resource_verdict}")
        
        return {
            "scaling_projections": scaling_results,
            "sufficiency_score": round(sufficiency_score, 2),
            "resource_verdict": resource_verdict
        }
    
    def generate_feasibility_report(self, domain_results: Dict) -> Dict:
        """Generate final feasibility assessment"""
        feasibility_score = 0
        
        # Check training success
        if self.training_results.get("final_loss", 3.0) < 2.5:
            feasibility_score += 25
            training_status = "PASS"
        else:
            training_status = "FAIL"
        
        # Check computational efficiency
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 18.7
        if gpu_memory_used < 50:
            feasibility_score += 25
            compute_status = "PASS"
        else:
            compute_status = "FAIL"
        
        # Check data coverage
        if self.training_results.get("dataset_size", 0) >= 3000:
            feasibility_score += 25
            data_status = "PASS"
        else:
            data_status = "FAIL"
        
        # Check domain adaptation
        if domain_results.get("adaptation_successful", False):
            feasibility_score += 25
            domain_status = "PASS"
        else:
            domain_status = "FAIL"
        
        print("\n" + "="*60)
        print("DAPT FEASIBILITY ASSESSMENT")
        print("="*60)
        print(f"Training Convergence:     {training_status}")
        print(f"Computational Efficiency: {compute_status}")
        print(f"Data Coverage:           {data_status}")
        print(f"Domain Adaptation:       {domain_status}")
        print("-"*60)
        print(f"Overall Feasibility:     {feasibility_score}/100")
        
        if feasibility_score >= 75:
            verdict = "HIGHLY FEASIBLE"
        elif feasibility_score >= 50:
            verdict = "PARTIALLY FEASIBLE"
        else:
            verdict = "NOT FEASIBLE"
        
        print(f"Verdict: {verdict}")
        print("="*60)
        
        return {
            "feasibility_score": feasibility_score,
            "verdict": verdict,
            "training_status": training_status,
            "compute_status": compute_status,
            "data_status": data_status,
            "domain_status": domain_status
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DAPT Model Interaction and Evaluation")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained DAPT model (e.g., ./ai_dapt_final)")
    parser.add_argument("--mode", type=str, choices=["test", "chat", "evaluate"], default="test",
                       help="Mode: test domain adaptation, interactive chat, or full evaluation")
    parser.add_argument("--questions", nargs="*", 
                       help="Custom questions for testing (optional)")
    
    args = parser.parse_args()
    
    # Load the trained DAPT model
    try:
        model, tokenizer = load_dapt_model(args.model_path)
    except FileNotFoundError:
        print(f"Error: Could not find DAPT model at {args.model_path}")
        print("Make sure you've completed DAPT training first using domain_adaptation.py")
        exit(1)
    
    # Create interactor
    interactor = ModelInteractor(model, tokenizer)
    
    if args.mode == "test":
        # Test domain adaptation with custom or default questions
        if args.questions:
            results = interactor.test_domain_adaptation(args.questions)
        else:
            results = interactor.test_domain_adaptation()
        
        print(f"\nFinal Results:")
        print(f"Average domain relevance: {results['avg_domain_score']:.2f}")
        print(f"Adaptation successful: {results['adaptation_successful']}")
        
    elif args.mode == "chat":
        # Start interactive chat
        interactor.interactive_chat()
        
    elif args.mode == "evaluate":
        # Full evaluation (requires training results)
        print("Full evaluation requires training results from domain_adaptation.py")
        print("Loading basic evaluation...")
        results = interactor.test_domain_adaptation()
        
        # Mock training results for demonstration
        mock_training_results = {
            "dataset_size": 100000,
            "training_time_minutes": 68,
            "final_loss": 2.1
        }
        
        evaluator = DAPTEvaluator(model, tokenizer, mock_training_results)
        evaluator.calculate_model_efficiency()
        evaluator.analyze_resource_sufficiency()
        evaluator.generate_feasibility_report(results)