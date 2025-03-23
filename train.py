import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from datetime import datetime
from model import CustomTransformerModel

# Import the CustomTransformerModel from your defined model code.
# from your_model_file import CustomTransformerModel

# For demonstration, we assume the CustomTransformerModel is defined in this script.
# (Please replace with your import if defined elsewhere.)

# -------------------------
# Configuration parameters
# -------------------------
MODEL_CONFIG = {
    "vocab_size": 50257,        # using GPT2 tokenizer vocab size
    "hidden_dim": 512,
    "num_layers": 6,
    "max_seq_len": 512,
    "pool_size": 10000,
    "tensor_dim": 2048,
    "conv_kernel_size": 3,
    "dilations": [1, 2, 4],
    "router_top_k": 2,
    "dropout": 0.1,
    "router_temperature": 0.5,
    "diversity_loss_scale": 0.01,
    "num_heads": 8  # unused in our attention-replacement design, but kept for compatibility
}

TRAINING_CONFIG = {
    "batch_size": 4,         # adjust based on your GPU memory
    "accumulation_steps": 4,
    "num_epochs": 3,
    "learning_rate": 5e-4,
    "log_interval": 50,      # steps between logging
    "sample_interval": 200,  # steps between sample generation
    "save_interval": 1000,   # steps between checkpoints
    "max_steps": 5000,
    "max_grad_norm": 1.0
}

OUTPUT_DIR = "checkpoint_lm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Data Preprocessing Helpers
# -------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # for padding

MAX_SEQ_LEN = MODEL_CONFIG["max_seq_len"]

def tokenize_function(example):
    # Tokenize the text and return input_ids.
    return tokenizer(example["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

def group_texts(examples):
    # Flatten and then group tokens into chunks of MAX_SEQ_LEN
    concatenated = sum(examples["input_ids"], [])
    total_length = len(concatenated)
    # We drop the last chunk if it doesn't fit exactly.
    total_length = (total_length // MAX_SEQ_LEN) * MAX_SEQ_LEN
    result = {
        "input_ids": [concatenated[i : i + MAX_SEQ_LEN] for i in range(0, total_length, MAX_SEQ_LEN)]
    }
    return result

# Create an iterable dataset using streaming
def get_streaming_dataset(split="train"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    # Filter out empty examples
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
    ds = ds.map(tokenize_function, batched=False)
    ds = ds.map(group_texts, batched=True)
    return ds

class LMIterableDataset(IterableDataset):
    def __init__(self, ds_iter):
        self.ds_iter = ds_iter

    def __iter__(self):
        for example in self.ds_iter:
            # Each example["input_ids"] is a list of token id lists.
            for ids in example["input_ids"]:
                # For LM, labels are shifted by one.
                # Here, we simply set input_ids as both inputs and labels.
                yield {"input_ids": torch.tensor(ids, dtype=torch.long),
                       "labels": torch.tensor(ids, dtype=torch.long)}

# -------------------------
# Initialize the Model
# -------------------------
# Instantiate your custom transformer model
model = CustomTransformerModel(
    vocab_size=MODEL_CONFIG["vocab_size"],
    hidden_dim=MODEL_CONFIG["hidden_dim"],
    num_layers=MODEL_CONFIG["num_layers"],
    max_seq_len=MODEL_CONFIG["max_seq_len"],
    pool_size=MODEL_CONFIG["pool_size"],
    tensor_dim=MODEL_CONFIG["tensor_dim"],
    conv_kernel_size=MODEL_CONFIG["conv_kernel_size"],
    dilations=MODEL_CONFIG["dilations"],
    router_top_k=MODEL_CONFIG["router_top_k"],
    dropout=MODEL_CONFIG["dropout"],
    router_temperature=MODEL_CONFIG["router_temperature"],
    diversity_loss_scale=MODEL_CONFIG["diversity_loss_scale"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

# -------------------------
# Training Loop
# -------------------------
def train():
    global_step = 0
    running_loss = 0.0
    accumulation_steps = TRAINING_CONFIG["accumulation_steps"]
    
    # Load dataset in streaming mode.
    ds = get_streaming_dataset(split="train")
    dataset = LMIterableDataset(ds)
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"])

    model.train()
    
    for epoch in range(1, TRAINING_CONFIG["num_epochs"] + 1):
        print(f"Epoch {epoch} started at {datetime.now()}")
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # LM: shift labels by one for prediction.
            # Here, we assume causal LM so input_ids serve as context and labels are shifted left.
            # For simplicity, we'll use input_ids as targets (ignoring the last token loss).
            outputs, div_loss = model(input_ids)
            # Shift outputs and labels appropriately.
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Compute cross-entropy loss.
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # Total loss includes diversity loss.
            total_loss = loss + div_loss
            
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            
            running_loss += total_loss.item()
            global_step += 1

            if global_step % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            # Log every log_interval steps.
            if global_step % TRAINING_CONFIG["log_interval"] == 0:
                avg_loss = running_loss / TRAINING_CONFIG["log_interval"]
                running_loss = 0.0
                mem_usage = torch.cuda.memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
                print(f"[Epoch {epoch} | Step {global_step}] Loss: {avg_loss:.4f} | Diversity Loss: {div_loss.item():.4f} | GPU Mem: {mem_usage:.1f} MB")

            # Sample output every sample_interval steps.
            if global_step % TRAINING_CONFIG["sample_interval"] == 0:
                sample_ids = input_ids[0:1]  # take first sequence in the batch
                # For a simple greedy decode, we feed the sample through the model.
                model.eval()
                with torch.no_grad():
                    sample_output, _ = model(sample_ids)
                # Take argmax and decode.
                generated_ids = torch.argmax(sample_output, dim=-1)[0].tolist()
                sample_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"=== Sample output at step {global_step} ===\n{sample_text}\n")
                model.train()

            # Save checkpoint every save_interval steps.
            if global_step % TRAINING_CONFIG["save_interval"] == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at step {global_step} to {ckpt_path}")

            if global_step >= TRAINING_CONFIG["max_steps"]:
                break
        if global_step >= TRAINING_CONFIG["max_steps"]:
            break

    # Save final model.
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    train()
