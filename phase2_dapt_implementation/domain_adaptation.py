#!/usr/bin/env python3
"""
Domain-Adaptive Pre-Training (DAPT) Engine for AI Research Papers
Handles training and model interaction for domain adaptation on AI abstracts.
"""

import torch
import pickle
import os
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Dict, List, Optional
import time


def load_processed_data(data_path: str) -> Dataset:
    """Load preprocessed and tokenized dataset from pickle file"""
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


class DAPTTrainer:
    """Domain-Adaptive Pre-Training trainer for AI research abstracts"""
    
    def __init__(self, model, tokenizer, output_dir: str = "./ai_dapt"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.trainer = None
        self.training_results = {}
        
    def setup_training(self, 
                      num_epochs: int = 3,
                      batch_size: int = 8,  # Increased for H200s
                      gradient_accumulation_steps: int = 2,  # Reduced since batch_size increased
                      learning_rate: float = 2e-5,
                      logging_steps: int = 100,
                      save_steps: int = 1000,
                      dataloader_num_workers: int = 4):  # Added for faster data loading
        """Configure training arguments optimized for 3 H200 GPUs"""
        
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=dataloader_num_workers,  # Faster data loading
            ddp_find_unused_parameters=False,  # Optimization for distributed training
        )
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
    
    def train(self, train_dataset: Dataset, resume_from_checkpoint: str = None) -> Dict:
        """Execute DAPT training with checkpoint recovery"""
        if self.training_args is None:
            self.setup_training()
            
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )
        
        # Check for existing checkpoints if not specified
        if resume_from_checkpoint is None:
            checkpoint_dir = self.output_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    # Get the latest checkpoint
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                    print(f"Found existing checkpoint: {resume_from_checkpoint}")
        
        print("Starting DAPT training...")
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        
        start_time = time.time()
        
        try:
            # Train the model
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            training_time = time.time() - start_time
            
            # Save the final model
            final_model_path = f"{self.output_dir}_final"
            self.trainer.save_model(final_model_path)
            print(f"DAPT complete! Model saved to {final_model_path}")
            
            # Collect training metrics
            self.training_results = {
                "training_time_minutes": training_time / 60,
                "final_loss": train_result.training_loss,
                "total_steps": self.trainer.state.global_step,
                "dataset_size": len(train_dataset),
                "model_path": final_model_path,
                "completed_successfully": True
            }
            
        except Exception as e:
            # Save current state before failing
            training_time = time.time() - start_time
            current_step = self.trainer.state.global_step if self.trainer else 0
            
            print(f"Training interrupted: {e}")
            print(f"Progress saved at step {current_step}")
            
            self.training_results = {
                "training_time_minutes": training_time / 60,
                "final_loss": None,
                "total_steps": current_step,
                "dataset_size": len(train_dataset),
                "model_path": None,
                "completed_successfully": False,
                "error": str(e)
            }
            
            raise e
        
        return self.training_results
    
    def get_training_metrics(self) -> Dict:
        """Get detailed training and efficiency metrics"""
        if not self.training_results:
            return {"error": "No training completed yet"}
        
        # Model specifications
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # GPU memory usage
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        
        metrics = {
            "training_feasibility": {
                "final_loss": round(self.training_results["final_loss"], 4),
                "total_steps": self.training_results["total_steps"],
                "dataset_size": self.training_results["dataset_size"],
                "training_time_minutes": round(self.training_results["training_time_minutes"], 2),
                "converged": self.training_results["final_loss"] < 2.5
            },
            "computational_efficiency": {
                "total_params": f"{total_params/1e6:.1f}M",
                "trainable_params": f"{trainable_params/1e6:.1f}M",
                "gpu_memory_used_gb": round(gpu_memory_used, 1),
                "gpu_utilization": f"{gpu_memory_used/gpu_memory_total:.1%}" if gpu_memory_total > 0 else "N/A"
            },
            "model_path": self.training_results["model_path"]
        }
        
        print("=== DAPT Feasibility Metrics ===")
        print(f"Training Convergence: Loss = {metrics['training_feasibility']['final_loss']} ({'Converged' if metrics['training_feasibility']['converged'] else 'Check'})")
        print(f"Computational Efficiency: {metrics['computational_efficiency']['gpu_memory_used_gb']}GB / {gpu_memory_total:.1f}GB GPU memory")
        print(f"Data Processing: {metrics['training_feasibility']['dataset_size']:,} abstracts")
        print(f"Model Scale: {metrics['computational_efficiency']['trainable_params']} trainable parameters")
        
        return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DAPT Training with Multi-GPU Support")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (auto-detect if not specified)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--target_batch_size", type=int, default=80, help="Target effective batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    
    args = parser.parse_args()
    
    # Example usage with multi-GPU auto-scaling
    from phase1_model_instantiation.model_setup import get_qwen_model  # Replace with actual import
    
    # Load model and tokenizer
    model, tokenizer = get_qwen_model()
    
    # Load processed data
    train_dataset = load_processed_data("data/qwen_processed_data.pkl")
    
    # Initialize trainer with GPU count
    trainer = DAPTTrainer(model, tokenizer, num_gpus=args.num_gpus)
    
    # Setup training with scaling
    trainer.setup_training(
        num_epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        target_batch_size=args.target_batch_size
    )
    
    # Train the model
    results = trainer.train(train_dataset, resume_from_checkpoint=args.resume)
    
    # Get detailed metrics
    metrics = trainer.get_training_metrics()
    print(f"Training completed with final loss: {results.get('final_loss', 'N/A')}")
    
    # Time estimates for different GPU counts
    print(f"\nTime estimates for 100K abstracts:")
    for gpus in [1, 3, 5]:
        estimated_time = 341 / gpus * 1.2  # 20% overhead
        print(f"  {gpus} GPU{'s' if gpus > 1 else ''}: ~{estimated_time/60:.1f} hours")