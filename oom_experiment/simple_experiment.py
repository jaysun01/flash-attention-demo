import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
import csv

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

########################################
# Dummy Dataset
########################################
class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return random token sequences
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        return tokens, tokens  # input, target identical for this dummy LM

########################################
# Multi-Head Attention with FlashAttention Option
########################################
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, flash=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.flash = flash and FLASH_AVAILABLE

    def forward(self, x):
        B, T, D = x.size()
        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim)

        if self.flash:
            # FlashAttention expects [B, T, H, D], FP16 or BF16
            # Ensure FP16
            assert q.dtype in [torch.float16, torch.bfloat16], "FlashAttention only supports fp16/bf16"
            out = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(),
                                  dropout_p=0.0, softmax_scale=None)
            # out is [B, T, H, D]
            out = out.reshape(B, T, self.num_heads * self.head_dim)
        else:
            # Standard attention: [B, T, H, D] -> [B, H, T, D] for compute
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
            # Convert back to [B, T, D]
            out = out.permute(0,2,1,3).contiguous().view(B, T, self.num_heads * self.head_dim)

        out = self.W_o(out)
        return out

########################################
# Transformer Block
########################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, flash=False):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads, flash=flash)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

########################################
# Simple Transformer Model
########################################
class SimpleTransformerModel(nn.Module):
    def __init__(self, dim, num_heads, depth, vocab_size, flash=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, flash=flash) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embedding(idx)
        # If you're using FlashAttention and want half precision in the model, cast here
        # The embedding output is float32 by default, safe to cast to half now.
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
    
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


########################################
# Memory Measurement Utilities
########################################
def get_memory_usage():
    # Return GPU memory allocated in MB
    return torch.cuda.memory_allocated() / (1024**2)

########################################
# Training Loop
########################################
def train(model, dataloader, optimizer, steps, device, use_half=False):
    model.train()
    for i, (inp, tgt) in enumerate(dataloader):
        if i >= steps:
            break
        # Move inputs and targets to device, keep them as integers
        inp, tgt = inp.to(device), tgt.to(device)

        # DO NOT cast inp or tgt to half. They are indices.
        # Just keep inp, tgt as is.
        
        optimizer.zero_grad()
        logits = model(inp)  # model will handle fp16 casting internally if needed
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()


########################################
# Main Experiment Logic
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--flash", action='store_true')
    parser.add_argument("--output_csv", type=str, default="results.csv")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = RandomDataset(num_samples=100, seq_length=args.seq_length, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    use_half = args.flash  # Use half precision if flash is enabled
    model = SimpleTransformerModel(
        dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        vocab_size=args.vocab_size,
        flash=args.flash
    ).to(device)
    if use_half:
        model = model.half()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_mem = get_memory_usage()
    time_start = time.time()

    success = True
    try:
        train(model, dataloader, optimizer, args.steps, device, use_half=use_half)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            success = False
        else:
            # Re-raise if it's another type of error
            raise e

    final_mem = get_memory_usage()
    time_end = time.time()

    mem_diff = final_mem - initial_mem
    elapsed_time = time_end - time_start

    # Append results to CSV
    fieldnames = ["batch_size", "seq_length", "model_dim", "num_heads", "depth", "flash", "success", "initial_mem_MB", "final_mem_MB", "mem_diff_MB", "elapsed_time_s"]
    write_header = False
    if args.output_csv:
        try:
            with open(args.output_csv, 'r') as f:
                pass
        except FileNotFoundError:
            write_header = True

    if args.output_csv:
        with open(args.output_csv, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "depth": args.depth,
                "flash": args.flash,
                "success": success,
                "initial_mem_MB": initial_mem,
                "final_mem_MB": final_mem,
                "mem_diff_MB": mem_diff,
                "elapsed_time_s": elapsed_time
            })

    if success:
        print("Training completed successfully.")
    else:
        print("OOM encountered. Training failed.")
