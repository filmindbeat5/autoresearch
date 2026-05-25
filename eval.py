"""Evaluation script for autoresearch models.

Runs evaluation on a trained checkpoint, computing perplexity and
optionally sampling from the model.
"""

import os
import math
import argparse
import torch
import tiktoken
from contextlib import nullcontext

from train import GPTConfig, GPT


def load_checkpoint(checkpoint_path: str, device: str):
    """Load a model checkpoint from disk."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model config from checkpoint
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    
    model = GPT(config)
    state_dict = checkpoint["model"]
    
    # Strip any unwanted prefix from state dict keys
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Loaded model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
    print(f"Checkpoint iter: {checkpoint.get('iter_num', 'unknown')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return model, config, checkpoint


@torch.no_grad()
def evaluate_loss(model, data_dir: str, split: str, block_size: int,
                  batch_size: int, eval_iters: int, device: str, ctx):
    """Evaluate model loss on a data split."""
    import numpy as np
    
    data_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    
    losses = []
    for _ in range(eval_iters):
        # Random batch
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


@torch.no_grad()
def sample(model, prompt: str, max_new_tokens: int, temperature: float,
           top_k: int, device: str, ctx):
    """Generate text from the model given a prompt."""
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    tokens = encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    return decode(y[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate an autoresearch model checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with train/val data")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of eval iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(42)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=ptdtype)
    
    model, config, checkpoint = load_checkpoint(args.checkpoint, args.device)
    
    # Evaluate on train and val splits
    for split in ["train", "val"]:
        data_path = os.path.join(args.data_dir, f"{split}.bin")
        if os.path.exists(data_path):
            loss, ppl = evaluate_loss(
                model, args.data_dir, split, config.block_size,
                args.batch_size, args.eval_iters, args.device, ctx
            )
            print(f"{split:5s} | loss: {loss:.4f} | perplexity: {ppl:.2f}")
    
    # Optional text generation
    if args.prompt is not None:
        print("\n--- Generated text ---")
        output = sample(model, args.prompt, args.max_new_tokens,
                        args.temperature, args.top_k, args.device, ctx)
        print(output)
        print("--- End of generation ---")


if __name__ == "__main__":
    main()
