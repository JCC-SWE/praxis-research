
def compute_metrics_batch(predictions, ground_truths):
    """Compute all metrics for a batch"""
    # F1 scores
    f1_scores = [compute_f1_score(pred, truth) for pred, truth in zip(predictions, ground_truths)]
    
    # ROUGE scores
    try:
        rouge_results = rouge_metric.compute(predictions=predictions, references=ground_truths)
        rouge_scores = rouge_results['rougeL'] if isinstance(rouge_results['rougeL'], list) else [rouge_results['rougeL']] * len(predictions)
    except:
        rouge_scores = [0.0] * len(predictions)
    
    # BLEU scores
    try:
        bleu_results = bleu_metric.compute(predictions=predictions, references=[[truth] for truth in ground_truths])
        bleu_scores = bleu_results['bleu'] if isinstance(bleu_results['bleu'], list) else [bleu_results['bleu']] * len(predictions)
    except:
        bleu_scores = [0.0] * len(predictions)
    
    return f1_scores, rouge_scores, bleu_scores
import os
import sys
import json
import csv
import warnings
import logging
from datetime import datetime as dt

# Suppress all noise
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)


def find_project_root(start, markers=("pyproject.toml", ".git", "README.md")):
    cur = os.path.abspath(start)
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            return None
        cur = nxt

PROJECT_ROOT = find_project_root(__file__)
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
azure_path = os.path.join(parent_dir, 'azure_resources')
blob_path = os.path.join(parent_dir, 'blob_interface')
sys.path.insert(0, azure_path)
sys.path.insert(0, blob_path)

# Import everything
try:
    from upload_to_blob import upload_to_blob
    from download_from_blob import download_blob
    from phase2_dapt_implementation.get_qa_data import get_abstracts_from_blob
    from get_qa_texts import pull_qa_texts
    from phase1_model_instantiation.func_test import _build_chat_prompt, generate_reply
    from phase1_model_instantiation.model_setup import get_qwen_model
    
    # Suppress evaluate loading
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import evaluate
        
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def compute_f1_score(prediction, ground_truth):
    """Fast F1 score computation"""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = len(pred_tokens & truth_tokens)
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)



# def save_results_to_csv(results, filename):
#     """Save results to CSV and upload to blob"""
#     # Save locally first
#     with open(filename, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)
    
#     # Upload to blob
#     try:
#         blob_path = f"evaluation_results/{filename}"
#         upload_to_blob(filename, blob_path)
#         print(f"Results uploaded to blob: {blob_path}")
#     except Exception as e:
#         print(f"Failed to upload to blob: {e}")
#         print(f"Results saved locally: {filename}")

# def print_summary_stats(results):
#     """Print summary statistics"""
#     metrics = ['f1_score', 'rouge_score', 'bleu_score']
    
#     print("\n" + "="*50)
#     print("EVALUATION SUMMARY") 
#     print("="*50)
    
#     for metric in metrics:
#         values = [r[metric] for r in results]
#         avg_score = sum(values) / len(values)
#         print(f"{metric.upper()}: {avg_score:.4f}")
    
#     print(f"\nTotal QA pairs evaluated: {len(results)}")
#     print("="*50)

import torch

def compute_perplexity(model, tokenizer, text):
    """
    Compute perplexity for a given text.
    
    Args:
        model: Your Qwen model (already loaded)
        tokenizer: Your Qwen tokenizer (already loaded)
        text: String to compute perplexity on (question, answer, or Q+A concatenated)
    
    Returns:
        float: Perplexity score
    """
    device = next(model.parameters()).device
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Perplexity is exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity

def save_results_to_csv(results, filename):
    """Save results to CSV and upload to blob"""
    # Save locally first
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Upload to blob
    try:
        blob_path = f"evaluation_results/{filename}"
        upload_to_blob(filename, blob_path)
        print(f"Results uploaded to blob: {blob_path}")
    except Exception as e:
        print(f"Failed to upload to blob: {e}")
        print(f"Results saved locally: {filename}")

def print_summary_stats(results):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY") 
    print("="*50)
    
    perplexities = [r['perplexity'] for r in results]
    avg_perplexity = sum(perplexities) / len(perplexities)
    print(f"AVERAGE PERPLEXITY: {avg_perplexity:.4f}")
    print(f"MIN PERPLEXITY: {min(perplexities):.4f}")
    print(f"MAX PERPLEXITY: {max(perplexities):.4f}")
    
    print(f"\nTotal QA pairs evaluated: {len(results)}")
    print("="*50)

if __name__ == "__main__":
    print("Starting perplexity evaluation...")

    
    # Load everything once at startup
    data_2023 = pull_qa_texts(data='qa-2023.txt')
    #data_2025 = pull_qa_texts(data='qa-2025.txt')
    model_path = "/workspace/praxis-research/base-model/qwen-2.5-3b/cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    model, tokenizer = get_qwen_model(model_path)
    
    # Choose which dataset to evaluate (change this as needed)
    qa_data = data_2023  # or data_2025
    
    # Handle data parsing
    if isinstance(qa_data, (bytes, bytearray)):
        qa_data = qa_data.decode("utf-8")
    if isinstance(qa_data, str):
        qa_data = json.loads(qa_data)
   
    print(f"Evaluating perplexity on {len(qa_data)} QA pairs...")    
    results = []
    predictions = []
    ground_truths = []
    questions = []
    
    batch_size = 50
    
    for i, qa_pair in enumerate(qa_data):
        question = qa_pair['question']
        ground_truth = qa_pair['answer']
        print(f'On the {i}th iteration')
        # Generate prediction using original working functions
        try:
            prompt = _build_chat_prompt(tokenizer, question)
            prediction = generate_reply(model, tokenizer, prompt)
        except Exception as e:
            prediction = f"Error: {str(e)[:50]}"

        print(f'On the {i}th iteration')
        
        # Compute perplexity on question+answer
        full_text = f"Question: {question}\nAnswer: {ground_truth}"
        perplexity = compute_perplexity(model, tokenizer, full_text)
        
        results.append({
            'qa_id': i + 1,
            'question': question,
            'ground_truth': ground_truth,
            'perplexity': perplexity
        })
    
    # Save and summarize
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"qa_perplexity_results_{timestamp}.csv"    
    save_results_to_csv(results, output_file)
    print_summary_stats(results)
    print(f"Results saved to {output_file}")
    print("Evaluation complete!")
