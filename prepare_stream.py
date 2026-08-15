"""
Prepara train.bin e val.bin com tokenização BPE sem estourar RAM.
Processa o corpus em streaming e escreve tokens em arquivo temporário.
"""
import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def list_valid_txt_files(data_dir):
    txt_files = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".txt"):
            continue
        if name.endswith(".raw.txt"):
            continue
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        txt_files.append(path)
    return txt_files


def main():
    parser = argparse.ArgumentParser(description="Prepara dados BPE em streaming para evitar OOM.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join(DATA_DIR, "bpe_vocab.json"),
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer_path):
        print(f"ERRO: tokenizador não encontrado: {args.tokenizer_path}")
        return

    txt_files = list_valid_txt_files(args.data_dir)
    if not txt_files:
        print(f"ERRO: nenhum .txt válido encontrado em {args.data_dir}")
        return

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    if vocab_size <= np.iinfo(np.uint16).max + 1:
        data_dtype = np.uint16
    elif vocab_size <= np.iinfo(np.uint32).max + 1:
        data_dtype = np.uint32
    else:
        print(f"ERRO: vocab_size {vocab_size} excede suporte de np.uint32")
        return

    tmp_tokens_path = os.path.join(args.data_dir, "all_tokens.tmp.bin")
    total_tokens = 0

    print(f"Arquivos usados ({len(txt_files)}):")
    with open(tmp_tokens_path, "wb") as tmpf:
        for path in txt_files:
            print(f"  - {os.path.basename(path)}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = tokenizer.encode(line).ids
                    if not ids:
                        continue
                    arr = np.asarray(ids, dtype=data_dtype)
                    arr.tofile(tmpf)
                    total_tokens += len(arr)

    if total_tokens == 0:
        print("ERRO: nenhum token gerado a partir dos arquivos de entrada")
        os.remove(tmp_tokens_path)
        return

    mm = np.memmap(tmp_tokens_path, dtype=data_dtype, mode="r", shape=(total_tokens,))
    n_train = int(args.train_ratio * total_tokens)
    train_data = mm[:n_train]
    val_data = mm[n_train:]

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    train_data.tofile(train_path)
    val_data.tofile(val_path)

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_path": os.path.abspath(args.tokenizer_path),
        "dtype": np.dtype(data_dtype).name,
        "files": [os.path.basename(p) for p in txt_files],
        "total_tokens": int(total_tokens),
    }
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\ntrain.bin: {os.path.getsize(train_path):,} bytes ({len(train_data):,} tokens)")
    print(f"val.bin:   {os.path.getsize(val_path):,} bytes ({len(val_data):,} tokens)")
    print(f"meta.json salvo em {meta_path}")

    print("\n=== SANITY CHECK ===")
    print(f"vocab_size final: {vocab_size}")
    print(f"total de tokens no corpus: {total_tokens:,}")
    decoded_slice = tokenizer.decode(train_data[:100].tolist())
    print("decode dos primeiros 100 IDs de treino:")
    print(decoded_slice)

    del mm
    os.remove(tmp_tokens_path)


if __name__ == "__main__":
    main()
