"# Extrinsic metrics" 
"""
TODO: Extrinsic (Task-Centric: Q&A, Abstract → Question/Answer)
6.	Exact Match (EM) – % of answers exactly matching ground truth.
7.	F1 Score – Token-level overlap between generated vs. reference answers.
8.	ROUGE (ROUGE-L, ROUGE-2) – Recall-oriented, good for abstractive answers.
9.	BLEU – Precision-oriented, checks n-gram overlap.
10.	METEOR / BERTScore – Embedding-based semantic similarity.
"""
import os
import sys
from datetime import datetime as dt
import os, sys

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

# Import everything at the top
from upload_to_blob import upload_to_blob
from download_from_blob import download_blob
from phase2_dapt_implementation.get_qa_data import get_abstracts_from_blob
from get_qa_texts import pull_qa_texts
from phase1_model_instantiation.func_test import _build_chat_prompt, generate_reply
from phase1_model_instantiation.model_setup import get_qwen_model

data_2023 = pull_qa_texts(data='qa-2023.txt')
data_2025 = pull_qa_texts(data='qa-2025.txt')
model,tokenizer = get_qwen_model()


import evaluate

def compute_f1_score(prediction, ground_truth):
    f1_metric = evaluate.load("f1")
    return f1_metric.compute(predictions=[prediction], references=[ground_truth])['f1']

def compute_rouge(prediction, ground_truth):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=[prediction], references=[ground_truth])['rougeL']

def compute_bleu(prediction, ground_truth):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=[prediction], references=[[ground_truth]])['bleu']

def compute_meteor(prediction, ground_truth):
    meteor = evaluate.load("meteor")
    return meteor.compute(predictions=[prediction], references=[ground_truth])['meteor']

import json

def load_qa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def generate_answer(model, tokenizer, question):
    prompt = _build_chat_prompt(tokenizer, question)
    return generate_reply(model, tokenizer, prompt)

def evaluate_model_on_qa(model, tokenizer, qa_data):
    results = []
    
    for i, qa_pair in enumerate(qa_data):
        question = qa_pair['question']
        ground_truth = qa_pair['answer']
        
        # Generate prediction
        prediction = generate_answer(model, tokenizer, question)
        
        # Compute metrics
        f1 = compute_f1_score(prediction, ground_truth)
        rouge = compute_rouge(prediction, ground_truth)
        bleu = compute_bleu(prediction, ground_truth)
        meteor = compute_meteor(prediction, ground_truth)
        
        results.append({
            'qa_id': i+1,
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'f1_score': f1,
            'rouge_score': rouge,
            'bleu_score': bleu,
            'meteor_score': meteor
        })
    
    return results

import csv

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# Save results
def evaluate_and_save():
    results_2023 = evaluate_model_on_qa(model, tokenizer, data_2023)
    results_2025 = evaluate_model_on_qa(model, tokenizer, data_2025)

    save_results_to_csv(results_2023, 'qwen_evaluation_2023.csv')
    save_results_to_csv(results_2025, 'qwen_evaluation_2025.csv')

if __name__ == "__main__":
    print("Starting evaluation...") 