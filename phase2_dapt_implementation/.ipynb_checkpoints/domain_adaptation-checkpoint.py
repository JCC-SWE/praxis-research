import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import pickle
from torch.nn.utils.rnn import pad_sequence

class CustomDataCollator:
    """Custom data collator that handles pre-tokenized data with different lengths."""
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Extract the lists from the dataset format
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # Pad sequences
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        attention_mask = (input_ids_padded != pad_id).long()

        # Pad to multiple of pad_to_multiple_of for efficiency
        if self.pad_to_multiple_of:
            max_len = input_ids_padded.size(1)
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                pad_len = self.pad_to_multiple_of - remainder
                batch_size = input_ids_padded.size(0)
                
                # Create padding tensors
                input_pad = torch.full((batch_size, pad_len), pad_id, dtype=torch.long)
                mask_pad = torch.zeros((batch_size, pad_len), dtype=torch.long)
                label_pad = torch.full((batch_size, pad_len), -100, dtype=torch.long)
                
                # Concatenate padding
                input_ids_padded = torch.cat([input_ids_padded, input_pad], dim=1)
                attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
                labels_padded = torch.cat([labels_padded, label_pad], dim=1)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

def load_processed_data(data_path):
    """Load preprocessed, tokenized HuggingFace Dataset + metadata from a pickle."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    dataset = data["dataset"]
    metadata = {
        "tokenizer_name": data["tokenizer_name"],
        "max_length": data["max_length"],
        "vocab_size": data["vocab_size"],
    }

    print(f"Loaded dataset: {len(dataset):,} examples")
    print(f"Tokenizer: {metadata['tokenizer_name']}")
    print(f"Max length: {metadata['max_length']}")
    print(f"Vocab size: {metadata['vocab_size']:,}")
    return dataset, metadata

def run_dapt_training(model, tokenizer, dataset, output_dir="./ai_dapt_final", resume_from_checkpoint=None):
    """
    Run Domain-Adaptive Pre-Training on the given model and dataset.
    
    Args:
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        dataset: HuggingFace Dataset with input_ids, attention_mask, labels
        output_dir: Where to save the final model
    
    Returns:
        Trained model path
    """
    
    # Dataset is already tokenized, just keep the training columns
    keep_columns = {'input_ids', 'attention_mask', 'labels'}
    cols_to_remove = [c for c in dataset.column_names if c not in keep_columns]
    if cols_to_remove:
        tokenized_dataset = dataset.remove_columns(cols_to_remove)
    else:
        tokenized_dataset = dataset

    # Use custom data collator for pre-tokenized data
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir="./ai_dapt_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=10,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting DAPT training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    print(f"DAPT complete! Model saved to {output_dir}")
    
    return output_dir