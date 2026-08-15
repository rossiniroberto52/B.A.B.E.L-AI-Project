import argparse
import json
import os

import torch

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description="Gera texto com um GPT treinado.")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    parser.add_argument("--checkpoint", type=str, default="ckpt.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.out_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(os.path.join(args.out_dir, "config.json")) as f:
        meta = json.load(f)

    if "tokenizer_path" in meta and meta["tokenizer_path"]:
        from tokenizers import Tokenizer as HFTokenizer

        tokenizer = HFTokenizer.from_file(meta["tokenizer_path"])
        encode = lambda s: tokenizer.encode(s).ids
        decode = lambda l: tokenizer.decode(l)
    else:
        chars = meta["chars"]
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join(itos[i] for i in l)

    if args.prompt:
        start = torch.tensor([encode(args.prompt)], dtype=torch.long)
    else:
        start = torch.zeros((1, 1), dtype=torch.long)

    out = model.generate(
        start,
        args.n_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
