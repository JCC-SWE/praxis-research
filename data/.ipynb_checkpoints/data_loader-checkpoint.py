import os
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataCollator:
    """Custom data collator for language modeling that handles padding properly."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features):
        # Extract sequences
        input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
        labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
        
        # Pad sequences to the same length
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss
        
        # Create attention mask
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = input_ids_padded.size(1)
            remainder = max_length % self.pad_to_multiple_of
            if remainder != 0:
                padding_length = self.pad_to_multiple_of - remainder
                pad_tensor = torch.full((input_ids_padded.size(0), padding_length), 
                                      self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids_padded = torch.cat([input_ids_padded, pad_tensor], dim=1)
                
                attention_pad = torch.zeros((attention_mask.size(0), padding_length), dtype=torch.long)
                attention_mask = torch.cat([attention_mask, attention_pad], dim=1)
                
                labels_pad = torch.full((labels_padded.size(0), padding_length), -100, dtype=torch.long)
                labels_padded = torch.cat([labels_padded, labels_pad], dim=1)
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded
        }

class QwenDataLoader:
    def __init__(self, processed_data_path="qwen_processed_data.pkl", 
                 train_split=0.7, val_split=0.2, test_split=0.1, 
                 batch_size=4, max_length=1024):
        """
        Initialize data loader for Qwen training with train/val/test splits.
        
        Args:
            processed_data_path (str): Path to processed pickle file
            train_split (float): Proportion for training 
            val_split (float): Proportion for validation
            test_split (float): Proportion for testing
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length
        """
        self.processed_data_path = processed_data_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Validate splits sum to 1.0
        if abs(train_split + val_split + test_split - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0. Got: {train_split + val_split + test_split}")
        
        # Load processed data
        self.dataset, self.metadata = self._load_processed_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata['tokenizer_name'])
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create train/val/test splits
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_splits()
        
        # Create custom data collator
        self.data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )
    
    def _load_processed_data(self):
        """Load the processed dataset from pickle file."""
        print(f"📂 Loading processed data from {self.processed_data_path}")
        
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Processed data file not found: {self.processed_data_path}")
        
        with open(self.processed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = data['dataset']
        metadata = {
            'tokenizer_name': data['tokenizer_name'],
            'max_length': data['max_length'],
            'vocab_size': data['vocab_size']
        }
        
        print(f"✅ Loaded {len(dataset)} samples")
        return dataset, metadata
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        total_size = len(self.dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size  # Remaining goes to test
        
        # Create non-overlapping splits
        train_dataset = self.dataset.select(range(train_size))
        val_dataset = self.dataset.select(range(train_size, train_size + val_size))
        test_dataset = self.dataset.select(range(train_size + val_size, total_size))
        
        # Remove non-tensor columns before training
        # Keep only the columns needed for language modeling
        columns_to_remove = [col for col in train_dataset.column_names 
                           if col not in ['input_ids', 'attention_mask', 'labels']]
        
        if columns_to_remove:
            print(f"🗑️ Removing metadata columns: {columns_to_remove}")
            train_dataset = train_dataset.remove_columns(columns_to_remove)
            val_dataset = val_dataset.remove_columns(columns_to_remove)
            test_dataset = test_dataset.remove_columns(columns_to_remove)
        
        # Fix any data type issues - ensure all fields are proper lists of integers
        def fix_data_types(examples):
            # Ensure input_ids and labels are flat lists of integers
            if 'input_ids' in examples:
                examples['input_ids'] = [ids if isinstance(ids, list) else ids.tolist() 
                                       for ids in examples['input_ids']]
            if 'labels' in examples:
                examples['labels'] = [labels if isinstance(labels, list) else labels.tolist() 
                                    for labels in examples['labels']]
            if 'attention_mask' in examples:
                examples['attention_mask'] = [mask if isinstance(mask, list) else mask.tolist() 
                                            for mask in examples['attention_mask']]
            return examples
        
        train_dataset = train_dataset.map(fix_data_types, batched=True)
        val_dataset = val_dataset.map(fix_data_types, batched=True)
        test_dataset = test_dataset.map(fix_data_types, batched=True)
        
        print(f"📊 Dataset splits:")
        print(f"  Training: {len(train_dataset)} samples ({self.train_split:.1%})")
        print(f"  Validation: {len(val_dataset)} samples ({self.val_split:.1%})")
        print(f"  Test: {len(test_dataset)} samples ({self.test_split:.1%})")
        print(f"  Columns: {train_dataset.column_names}")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_train_dataloader(self, shuffle=True):
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=0,  # Set to 0 for Azure ML compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_val_dataloader(self, shuffle=False):
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_test_dataloader(self, shuffle=False):
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_sample_batch(self, split='train'):
        """Get a sample batch for testing."""
        if split == 'train':
            dataloader = self.get_train_dataloader()
        elif split == 'val':
            dataloader = self.get_val_dataloader()
        elif split == 'test':
            dataloader = self.get_test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        batch = next(iter(dataloader))
        return batch
    
    def print_batch_info(self, batch):
        """Print information about a batch."""
        print(f"\n📦 Batch Information:")
        print(f"Batch size: {batch['input_ids'].shape[0]}")
        print(f"Sequence length: {batch['input_ids'].shape[1]}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Show token statistics for this batch
        seq_lengths = (batch['attention_mask'].sum(dim=1)).tolist()
        print(f"Actual sequence lengths in batch: {seq_lengths}")
        print(f"Average sequence length: {sum(seq_lengths) / len(seq_lengths):.1f}")
    
    def decode_sample(self, input_ids, max_tokens=100):
        """Decode a sample to see the actual text."""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        # Truncate for readability
        truncated_ids = input_ids[:max_tokens]
        decoded_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
        return decoded_text
    
    def get_dataset_info(self):
        """Get comprehensive dataset information."""
        info = {
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'tokenizer': self.metadata['tokenizer_name'],
            'vocab_size': self.metadata['vocab_size']
        }
        return info

def test_dataloader(processed_data_path="qwen_processed_data.pkl"):
    """Test the data loader functionality."""
    print("🧪 Testing DataLoader...")
    
    # Initialize data loader with train/val/test splits
    loader = QwenDataLoader(
        processed_data_path=processed_data_path,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        batch_size=2,  # Small batch for testing
        max_length=1024
    )
    
    # Print dataset info
    info = loader.get_dataset_info()
    print(f"\n📊 Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test train dataloader
    print(f"\n🔄 Testing train dataloader...")
    train_batch = loader.get_sample_batch('train')
    loader.print_batch_info(train_batch)
    
    # Show sample decoded text
    print(f"\n📝 Sample decoded text (first 200 chars):")
    sample_text = loader.decode_sample(train_batch['input_ids'][0])
    print(f"{sample_text[:200]}...")
    
    # Test val dataloader
    print(f"\n🔄 Testing validation dataloader...")
    val_batch = loader.get_sample_batch('val')
    loader.print_batch_info(val_batch)
    
    # Test test dataloader
    print(f"\n🔄 Testing test dataloader...")
    test_batch = loader.get_sample_batch('test')
    loader.print_batch_info(test_batch)
    
    print(f"\n✅ DataLoader test complete!")
    return loader

if __name__ == "__main__":
    # Test the data loader
    loader = test_dataloader()