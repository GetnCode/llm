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
from typing import Tuple, List, Optional

# Global debug flag. Set to False to disable debug outputs.
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def print_tensor_stats(name: str, tensor: torch.Tensor):
    if not DEBUG or tensor.numel() == 0:
        return
    debug_print(f"{name}: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")

########################################
# 1. Dilated Hierarchical Convolutions (Local Branch)
########################################

class DilatedHierarchicalConvolution(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilations: List[int], dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            padding = ((kernel_size - 1) // 2) * dilation
            self.layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation))
        self.out_proj = nn.Linear(hidden_dim * len(dilations), hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be (B, seq_len, hidden_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_trans = x.transpose(1, 2)  # (B, hidden_dim, seq_len)
        conv_outs = []
        for i, conv in enumerate(self.layers):
            out = conv(x_trans)
            out = self.activation(out)
            print_tensor_stats(f"DilatedConv layer {i} output", out)
            conv_outs.append(out.transpose(1, 2))
        concat_out = torch.cat(conv_outs, dim=-1)
        proj = self.out_proj(concat_out)
        out = self.dropout(proj)
        print_tensor_stats("DilatedHierarchicalConvolution final output", out)
        return out

########################################
# 2. Shared Static Tensor Pool (Memory)
########################################

class StaticTensorPool(nn.Module):
    def __init__(self, pool_size: int, tensor_dim: int):
        super().__init__()
        pool = torch.randn(pool_size, tensor_dim)
        self.register_buffer('pool', pool / pool.norm(dim=-1, keepdim=True))
        
    def get_pool(self) -> torch.Tensor:
        return self.pool

########################################
# 3. AttentionRouter (Transformation Router) with Debug Logging
########################################

class AttentionRouter(nn.Module):
    def __init__(self, hidden_dim: int, pool_size: int, tensor_dim: int, 
                 top_k: int = 2, temperature: float = 0.5, diversity_loss_scale: float = 0.01):
        """
        Args:
            hidden_dim: Input token dimension.
            pool_size: Number of entries in the tensor pool.
            tensor_dim: Dimensionality for pool entries and projection.
            top_k: Number of pool entries to select.
            temperature: Initial temperature for softmax scaling.
            diversity_loss_scale: Scaling factor for diversity loss.
        """
        super().__init__()
        self.intermediate_dim = min(1024, pool_size)
        self.linear1 = nn.Linear(hidden_dim, self.intermediate_dim)
        self.linear2 = nn.Linear(self.intermediate_dim, pool_size)
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.tensor_dim = tensor_dim
        self.input_proj = nn.Linear(hidden_dim, tensor_dim)
        self.mapping = nn.Linear(2 * tensor_dim, tensor_dim)
        self.diversity_loss_scale = diversity_loss_scale
        self.norm_proj = nn.LayerNorm(tensor_dim)

        # Backward hook for mapping layer (only prints if DEBUG is True)
        self.mapping.register_backward_hook(lambda module, grad_input, grad_output:
            debug_print("Mapping grad stats:", [(f"min {g.min().item():.6f}, max {g.max().item():.6f}") 
                                                  for g in grad_input if g is not None]))

    def forward(self, x: torch.Tensor, tensor_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq_len, _ = x.shape
        inter = F.relu(self.linear1(x))
        print_tensor_stats("AttentionRouter linear1 output", inter)
        logits = self.linear2(inter)
        print_tensor_stats("AttentionRouter logits", logits)
        temp = self.temperature.clamp(min=0.1, max=5.0)
        scaled_logits = torch.clamp(logits / temp, -10, 10)
        print_tensor_stats("Scaled logits", scaled_logits)
        topk_logits, topk_indices = scaled_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        print_tensor_stats("Top-k weights", topk_weights)
        experts_out = tensor_pool[topk_indices]  # shape: (B, seq_len, top_k, tensor_dim)
        weighted_map = torch.einsum('bsk,bskd->bsd', topk_weights, experts_out)
        print_tensor_stats("Weighted map", weighted_map)
        projected_x = self.norm_proj(self.input_proj(x))
        print_tensor_stats("Projected x", projected_x)
        combined = torch.cat([projected_x, weighted_map], dim=-1)
        transformation = self.mapping(combined)
        print_tensor_stats("Transformation output", transformation)
        
        # Compute diversity loss.
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)
        usage = torch.zeros(tensor_pool.size(0), device=x.device).scatter_add(0, flat_indices, flat_weights)
        usage_fraction = usage / (usage.sum() + 1e-8)
        uniform_target = torch.full_like(usage_fraction, 1.0 / tensor_pool.size(0))
        diversity_loss = F.mse_loss(usage_fraction, uniform_target) * min(1.0, x.numel() / (tensor_pool.size(0) * self.top_k))
        diversity_loss = diversity_loss * self.diversity_loss_scale
        
        return transformation, diversity_loss

########################################
# 4. GlobalLocalAttention Module with Debug Logging
########################################

class GlobalLocalAttention(nn.Module):
    def __init__(self, hidden_dim: int, pool_size: int, tensor_dim: int,
                 conv_kernel_size: int, dilations: List[int],
                 router_top_k: int = 2, router_temperature: float = 0.5,
                 dropout: float = 0.0, diversity_loss_scale: float = 0.01):
        super().__init__()
        self.conv_module = DilatedHierarchicalConvolution(hidden_dim, conv_kernel_size, dilations, dropout)
        self.router = AttentionRouter(hidden_dim, pool_size, tensor_dim, top_k=router_top_k, 
                                      temperature=router_temperature, diversity_loss_scale=diversity_loss_scale)
        self.router_proj = nn.Linear(tensor_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tensor_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        local_out = self.conv_module(x)
        print_tensor_stats("Local branch output", local_out)
        router_out, attn_div_loss = self.router(x, tensor_pool)
        global_out = self.router_proj(router_out)
        print_tensor_stats("Global branch output", global_out)
        gate = torch.sigmoid(self.gate_proj(x))
        print_tensor_stats("Gate output", gate)
        attn_out = gate * local_out + (1 - gate) * global_out
        attn_out = self.dropout(attn_out)
        print_tensor_stats("GlobalLocalAttention final output", attn_out)
        return attn_out, attn_div_loss

########################################
# 5. Custom Transformer Block with Debug Logging
########################################

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, pool_size: int, tensor_dim: int,
                 conv_kernel_size: int, dilations: List[int], router_top_k: int = 2,
                 dropout: float = 0.0, router_temperature: float = 0.5,
                 diversity_loss_scale: float = 0.01):
        super().__init__()
        self.attn_replacement = GlobalLocalAttention(hidden_dim, pool_size, tensor_dim,
                                                      conv_kernel_size, dilations, router_top_k,
                                                      router_temperature, dropout, diversity_loss_scale)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn_router = AttentionRouter(hidden_dim, pool_size, tensor_dim,
                                          top_k=router_top_k, temperature=router_temperature,
                                          diversity_loss_scale=diversity_loss_scale)
        self.router_proj = nn.Linear(tensor_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, tensor_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_div_loss = self.attn_replacement(x, tensor_pool)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out, ffn_div_loss = self.ffn_router(x, tensor_pool)
        x = self.norm2(x + self.dropout(self.router_proj(ffn_out)))
        x = self.norm_ffn(x + self.ffn(x))
        total_div_loss = attn_div_loss + ffn_div_loss
        if torch.isnan(x).any():
            debug_print("NaN detected after transformer block")
        return x, total_div_loss

########################################
# 6. Complete Transformer Model with Debug Logging
########################################

class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, max_seq_len: int,
                 pool_size: int = 10000, tensor_dim: int = None, conv_kernel_size: int = 3,
                 dilations: List[int] = [1, 2, 4], router_top_k: int = 2, dropout: float = 0.0,
                 router_temperature: float = 0.5, diversity_loss_scale: float = 0.01):
        super().__init__()
        if tensor_dim is None:
            tensor_dim = 4 * hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.shared_tensor_pool = StaticTensorPool(pool_size, tensor_dim)
        self.layers = nn.ModuleList([
            CustomTransformerBlock(hidden_dim, pool_size, tensor_dim, conv_kernel_size,
                                   dilations, router_top_k, dropout, router_temperature, diversity_loss_scale)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)  # (B, seq_len, hidden_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        total_div_loss = 0.0
        tensor_pool = self.shared_tensor_pool.get_pool()
        for layer in self.layers:
            x, div_loss = layer(x, tensor_pool)
            total_div_loss += div_loss
        x = self.norm(x)
        logits = self.output_head(x)
        if torch.isnan(logits).any():
            debug_print("NaN detected in final logits")
        return logits, total_div_loss * 0.01

########################################
# 7. Training Script for LM with Streaming Dataset and Debug Logging
########################################

MODEL_CONFIG = {
    "vocab_size": 50257,        # GPT-2 vocab size
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
    "diversity_loss_scale": 0.01
}

TRAINING_CONFIG = {
    "batch_size": 4,
    "accumulation_steps": 4,
    "num_epochs": 3,
    "learning_rate": 5e-4,
    "log_interval": 50,
    "sample_interval": 200,
    "save_interval": 1000,
    "max_steps": 5000,
    "max_grad_norm": 1.0
}

OUTPUT_DIR = "checkpoint_lm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
MAX_SEQ_LEN = MODEL_CONFIG["max_seq_len"]

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = len(concatenated)
    total_length = (total_length // MAX_SEQ_LEN) * MAX_SEQ_LEN
    result = {"input_ids": [concatenated[i : i + MAX_SEQ_LEN] for i in range(0, total_length, MAX_SEQ_LEN)]}
    return result

def get_streaming_dataset(split="train"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
    ds = ds.map(tokenize_function, batched=False)
    ds = ds.map(group_texts, batched=True)
    return ds

class LMIterableDataset(IterableDataset):
    def __init__(self, ds_iter):
        self.ds_iter = ds_iter

    def __iter__(self):
        for example in self.ds_iter:
            for ids in example["input_ids"]:
                yield {"input_ids": torch.tensor(ids, dtype=torch.long),
                       "labels": torch.tensor(ids, dtype=torch.long)}

def collate_fn(batch):
    return {key: torch.stack([d[key] for d in batch], dim=0) for key in batch[0]}

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

def train():
    global_step = 0
    running_loss = 0.0
    accumulation_steps = TRAINING_CONFIG["accumulation_steps"]
    ds = get_streaming_dataset(split="train")
    dataset = LMIterableDataset(ds)
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], collate_fn=collate_fn)
    model.train()
    
    for epoch in range(1, TRAINING_CONFIG["num_epochs"] + 1):
        debug_print(f"Epoch {epoch} started at {datetime.now()}")
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if input_ids.dim() != 2 or labels.dim() != 2:
                debug_print("Warning: Unexpected shapes:", input_ids.shape, labels.shape)
            
            # Check for problematic batches (e.g. only pad tokens)
            unique_labels = torch.unique(labels)
            if unique_labels.numel() == 1 and unique_labels.item() == tokenizer.pad_token_id:
                debug_print("Warning: Batch contains only pad tokens. Skipping this batch.")
                continue
            else:
                debug_print("Unique label IDs in batch:", unique_labels.tolist())
            
            outputs, div_loss = model(input_ids)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            print_tensor_stats("Shifted logits", shift_logits)
            print_tensor_stats("Shifted labels", shift_labels.float())
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss = loss + div_loss
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            running_loss += total_loss.item()
            global_step += 1

            if global_step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            if global_step % TRAINING_CONFIG["log_interval"] == 0:
                avg_loss = running_loss / TRAINING_CONFIG["log_interval"]
                running_loss = 0.0
                mem_usage = torch.cuda.memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
                print(f"[Epoch {epoch} | Step {global_step}] Loss: {avg_loss:.4f} | Diversity Loss: {div_loss.item():.4f} | GPU Mem: {mem_usage:.1f} MB")

            if global_step % TRAINING_CONFIG["sample_interval"] == 0:
                sample_ids = input_ids[0:1]
                model.eval()
                with torch.no_grad():
                    sample_output, _ = model(sample_ids)
                generated_ids = torch.argmax(sample_output, dim=-1)[0].tolist()
                sample_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"=== Sample output at step {global_step} ===\n{sample_text}\n")
                model.train()

            if global_step % TRAINING_CONFIG["save_interval"] == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at step {global_step} to {ckpt_path}")

            if global_step >= TRAINING_CONFIG["max_steps"]:
                break
        if global_step >= TRAINING_CONFIG["max_steps"]:
            break

    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    train()
