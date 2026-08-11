import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, GPTConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def physical_cores():
    try:
        cores = set()
        for name in os.listdir("/sys/devices/system/cpu"):
            if name.startswith("cpu") and name[3:].isdigit():
                with open(f"/sys/devices/system/cpu/{name}/topology/core_id") as f:
                    cores.add(f.read().strip())
        return len(cores) if cores else None
    except OSError:
        return None


def get_batch(split, data, block_size, batch_size, train_split, device):
    split_idx = int(train_split * len(data))
    if split == "train":
        data_slice = data[:split_idx]
    else:
        data_slice = data[split_idx:]
    n = len(data_slice)
    ix = torch.randint(n - block_size - 1, (batch_size,))
    x = torch.stack([data_slice[i : i + block_size] for i in ix])
    y = torch.stack([data_slice[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def evaluate(model, data, args, config, device):
    model.eval()
    losses = []
    for split in ("train", "val"):
        loss_accum = 0.0
        n_steps = max(1, args.eval_iters // args.batch_size)
        for _ in range(n_steps):
            x, y = get_batch(
                split, data, config.block_size, args.batch_size, args.train_split, device,
            )
            with torch.no_grad():
                _, loss = model(x, y)
            loss_accum += loss.item()
        losses.append(loss_accum / n_steps)
    model.train()
    return tuple(losses)


@torch.no_grad()
def sample(model, tokenizer, n_tokens, temperature, top_k, device):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, n_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(model, config, best_val_loss, iter_num, path):
    ckpt = {
        "model": model.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "iter_num": iter_num,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Treina um GPT do zero.")
    parser.add_argument("--dataset", type=str, default="tinyshakespeare")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--sample_n_tokens", type=int, default=256)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=4, help="evals sem melhora antes de parar")
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--compile", action="store_true", help="usa torch.compile")

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads <= 0:
        args.threads = physical_cores() or os.cpu_count() or 1
    torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda pedido mas indisponível; usando cpu")
        device = "cpu"

    data_path = os.path.join(args.data_dir, f"{args.dataset}.txt")
    if not os.path.exists(data_path):
        print(f"dataset não encontrado em {data_path}")
        sys.exit(1)

    text = open(data_path, encoding="utf-8").read()

    chars = sorted(list(set(text)))
    vocab_size = args.vocab_size or len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "encode": lambda self, s, stoi=stoi: [stoi[c] for c in s],
            "decode": lambda self, l, itos=itos: "".join(itos[i] for i in l),
        },
    )()

    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    data = np.array([stoi[chr(b)] for b in data], dtype=np.int64)
    data_t = torch.from_numpy(data)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "block_size": args.block_size,
                "vocab_size": vocab_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
                "chars": chars,
            },
            f,
        )
    with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config)
    model.to(device)
    if args.compile:
        model = torch.compile(model)
    n_params = model.num_parameters()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "—"
    print(f"dataset: {args.dataset} | tokens: {len(data):,} | vocab: {vocab_size}")
    print(f"modelo: {n_params/1e6:.2f}M params | block_size={config.block_size}")
    print(f"device: {device} {gpu_name} | threads: {torch.get_num_threads()}")
    mb = args.batch_size * config.block_size * 3 * 4 / 1e6
    print(f"bytes/step (fwd+bwd): {mb:.1f} MB x grad_accum={args.grad_accum}")

    lr = args.lr
    def get_lr(it):
        if it < args.warmup_iters:
            return lr * (it + 1) / (args.warmup_iters + 1)
        if it > args.max_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        return args.min_lr + 0.5 * (lr - args.min_lr) * (1 + math.cos(math.pi * decay_ratio))

    param_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    grad_norm_accum = 0.0

    best_val_loss = float("inf")
    bad_evals = 0
    early_stopped = False
    t0 = time.time()

    for it in range(args.max_iters):
        if it % args.eval_interval == 0:
            train_loss, val_loss = evaluate(model, data_t, args, config, device)
            elapsed = time.time() - t0
            print(
                f"iter {it:6d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {get_lr(it):.2e} | {elapsed:6.1f}s"
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                bad_evals = 0
                save_checkpoint(
                    model, config, best_val_loss, it, os.path.join(args.out_dir, "ckpt.pt")
                )
            else:
                bad_evals += 1
            print(f"        checkpoint {'salvo (melhor val)' if improved else 'mantido'}")
            if not args.no_early_stop and bad_evals >= args.patience:
                print(
                    f"early stop no iter {it} (val sem melhora por {args.patience} avaliações; "
                    f"melhor val {best_val_loss:.4f})"
                )
                early_stopped = True
                break

        if it % args.sample_interval == 0 and it > 0:
            text_sample = sample(
                model,
                tokenizer,
                args.sample_n_tokens,
                args.sample_temperature,
                args.sample_top_k,
                device,
            )
            print(f"---- amostra iter {it} ----")
            print(text_sample)
            print("--------------------------")
            with open(os.path.join(args.out_dir, "samples.txt"), "a", encoding="utf-8") as f:
                f.write(f"===== iter {it} =====\n{text_sample}\n\n")

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(
                "train", data_t, config.block_size, args.batch_size, args.train_split, device,
            )
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_accum += loss.item()
        if args.grad_clip > 0.0:
            grad_norm_accum = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for g in optimizer.param_groups:
            g["lr"] = get_lr(it)
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(
                f"iter {it:6d} | loss {loss_accum/args.grad_accum:.4f} | "
                f"gnorm {grad_norm_accum:.3f} | {dt:.1f}s"
            )

    train_loss, val_loss = evaluate(model, data_t, args, config, device)
    save_checkpoint(
        model, config, best_val_loss, it, os.path.join(args.out_dir, "ckpt_final.pt")
    )
    print(
        f"fim: train {train_loss:.4f} | val {val_loss:.4f} | melhor val {best_val_loss:.4f}"
        f"{' | early stop' if early_stopped else ''}"
    )
    print(f"checkpoints em {args.out_dir}")


if __name__ == "__main__":
    main()
