import sys
import os
from datetime import datetime

# Add path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'azure_resources'))

# Import data processing modules
from .cosmos_client import get_nlp_ready_data
from .preprocessing import preprocess_for_qwen, get_dataset_stats
from .data_loader import QwenDataLoader

def create_data_loaders(container_name="s_scholar_container", 
                       model_name="Qwen/Qwen2.5-3B", 
                       max_length=1024,
                       batch_size=4,
                       train_split=0.7,
                       val_split=0.2,
                       test_split=0.1):
    """
    Create train, validation, and test data loaders from CosmosDB data.
    
    Args:
        container_name (str): CosmosDB container to read from
        model_name (str): HuggingFace model name for tokenizer
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for data loaders
        train_split (float): Training split ratio
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n📊 Creating Train/Val/Test Data Loaders from CosmosDB")
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0. Got: {train_split + val_split + test_split}")
    
    # 1. Get data from CosmosDB
    print(f"📂 Fetching data from {container_name}...")
    data = get_nlp_ready_data(container_name)
    
    if not data:
        raise ValueError(f"No data found in {container_name}")
    
    print(f"📄 Found {len(data)} records")
    
    # 2. Preprocess for Qwen
    print("🔄 Preprocessing data for Qwen...")
    dataset = preprocess_for_qwen(
        data, 
        model_name=model_name,
        max_length=max_length,
        save_path="qwen_processed_data.pkl"
    )
    
    # 3. Show dataset statistics
    get_dataset_stats(dataset)
    
    # 4. Create data loaders with train/val/test splits
    print("📦 Creating train/val/test data loaders...")
    loader = QwenDataLoader(
        processed_data_path="qwen_processed_data.pkl",
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        batch_size=batch_size
    )
    
    train_loader = loader.get_train_dataloader()
    val_loader = loader.get_val_dataloader()
    test_loader = loader.get_test_dataloader()
    
    print(f"✅ Data loaders created:")
    print(f"   Training batches: {len(train_loader)} ({train_split:.1%})")
    print(f"   Validation batches: {len(val_loader)} ({val_split:.1%})")
    print(f"   Test batches: {len(test_loader)} ({test_split:.1%})")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader

def test_data_sample(container_name="s_scholar_container"):
    """Test function to examine data sample."""
    print("\n🧪 Testing Data Sample")
    
    # Get sample data
    data = get_nlp_ready_data(container_name)
    if data:
        sample = data[0]
        print(f"\nSample record:")
        print(f"ID: {sample['id']}")
        print(f"Title: {sample['title']}")
        print(f"Authors: {len(sample['authors'])} authors")
        print(f"Publication Year: {sample['publication_year']}")
        print(f"Venue: {sample['venue']}")
        print(f"Source: {sample['source']}")
        print(f"Text preview: {sample['text'][:200]}...")
    else:
        print("❌ No data found!")

if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Test data sample
    test_data_sample()
    
    # Option 2: Create data loaders
    # train_loader, val_loader, test_loader = create_data_loaders()
    
    # Option 3: Custom parameters
    # train_loader, val_loader, test_loader = create_data_loaders(
    #     container_name="s_scholar_container",
    #     batch_size=8,
    #     train_split=0.8,
    #     val_split=0.15,
    #     test_split=0.05
    # )