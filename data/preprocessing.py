import sys
import os
import re
import json
import pickle
from transformers import AutoTokenizer
from datasets import Dataset
import torch

# Add path for cosmos_client
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'azure_resources'))
from cosmos_client import get_nlp_ready_data, get_combined_nlp_data

def clean_text(text):
    """
    Basic text cleaning for academic/research content.
    
    Args:
        text (str): Raw text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_for_qwen(data, model_name="Qwen/Qwen2.5-3B-Instruct", max_length=2048, save_path="processed_data.pkl"):
    """
    Preprocess data for Qwen SLM training using HuggingFace tokenizers.
    
    Args:
        data (list): Raw data from cosmos_client
        model_name (str): HuggingFace model name for tokenizer
        max_length (int): Maximum sequence length
        save_path (str): Path to save processed data
    
    Returns:
        Dataset: HuggingFace Dataset with tokenized data
    """
    print(f"🔄 Preprocessing {len(data)} records for {model_name}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Clean and prepare text
    cleaned_data = []
    for record in data:
        cleaned_text = clean_text(record.get('text', ''))
        
        # Skip very short texts
        if len(cleaned_text) < 50:
            continue
        
        # Create training format (for causal LM)
        processed_record = {
            'text': cleaned_text,
            'id': record.get('id'),
            'title': record.get('title', ''),
            'source': record.get('source', ''),
            'publication_year': record.get('publication_year')
        }
        cleaned_data.append(processed_record)
    
    print(f"✅ Cleaned data: {len(cleaned_data)} records")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(cleaned_data)
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Don't pad here, will pad in data loader
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Apply tokenization
    print("🔄 Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Remove original text, keep metadata
    )
    
    print(f"✅ Tokenized {len(tokenized_dataset)} samples")
    
    # Save processed dataset
    print(f"💾 Saving processed data to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'dataset': tokenized_dataset,
            'tokenizer_name': model_name,
            'max_length': max_length,
            'vocab_size': tokenizer.vocab_size
        }, f)
    
    return tokenized_dataset

def load_processed_data(save_path="processed_data.pkl"):
    """Load previously processed data."""
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['dataset'], data

def get_dataset_stats(dataset):
    """Print statistics about the processed dataset."""
    print(f"\n📊 Dataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    # Token length statistics
    token_lengths = [len(sample['input_ids']) for sample in dataset]
    print(f"Average tokens per sample: {sum(token_lengths) / len(token_lengths):.1f}")
    print(f"Max tokens: {max(token_lengths)}")
    print(f"Min tokens: {min(token_lengths)}")
    
    # Source distribution
    sources = [sample['source'] for sample in dataset]
    source_counts = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} samples")

if __name__ == "__main__":
    # Get data from semantic scholar container only
    print("📊 Fetching data from s_scholar_container...")
    data = get_nlp_ready_data('s_scholar_container')
    
    if data:
        print(f"📄 Found {len(data)} records from Semantic Scholar")
        
        # Preprocess for Qwen
        dataset = preprocess_for_qwen(
            data, 
            model_name="Qwen/Qwen2.5-3B",
            max_length=1024,  # Adjust based on your needs
            save_path="qwen_processed_data.pkl"
        )
        
        # Show statistics
        get_dataset_stats(dataset)
        
        print(f"\n✅ Preprocessing complete! Dataset ready for training.")
    else:
        print("❌ No data found in s_scholar_container!")