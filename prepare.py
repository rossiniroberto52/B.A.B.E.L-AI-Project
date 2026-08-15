"""
Gera train.bin e val.bin com tokenização BPE treinada previamente.
Lê todos os arquivos .txt válidos em data/ e codifica o corpus completo.
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
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                sample = f.read(1024)
            if sample.strip():
                txt_files.append(path)
        except (OSError, UnicodeDecodeError):
            continue
    return txt_files


def main():
    parser = argparse.ArgumentParser(description="Prepara dados BPE para treino.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join(DATA_DIR, "bpe_vocab.json"),
        help="Caminho do tokenizador BPE treinado",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Proporção de treino (resto é validação)")
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

    all_token_ids = []
    total_chars = 0

    print(f"Arquivos usados ({len(txt_files)}):")
    for path in txt_files:
        print(f"  - {os.path.basename(path)}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        total_chars += len(text)
        encoded = tokenizer.encode(text).ids
        all_token_ids.extend(encoded)

    data = np.array(all_token_ids, dtype=data_dtype)

    # --- split treino / validação ---
    n = int(args.train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # --- salva como binários ---
    out_dir = args.data_dir
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    train_data.tofile(train_path)
    val_data.tofile(val_path)

    print(f"\ntrain.bin: {os.path.getsize(train_path):,} bytes ({len(train_data):,} tokens)")
    print(f"val.bin:   {os.path.getsize(val_path):,} bytes ({len(val_data):,} tokens)")

    # --- salva metadados do vocabulário ---
    meta = {
        "vocab_size": vocab_size,
        "tokenizer_path": os.path.abspath(args.tokenizer_path),
        "dtype": np.dtype(data_dtype).name,
        "files": [os.path.basename(p) for p in txt_files],
        "total_chars": total_chars,
        "total_tokens": int(len(data)),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"meta.json salvo em {meta_path}")

    # --- sanity check ---
    print("\n=== SANITY CHECK ===")
    print(f"vocab_size final: {vocab_size}")
    print(f"total de tokens no corpus: {len(data):,}")
    decoded_slice = tokenizer.decode(train_data[:100].tolist())
    print("decode dos primeiros 100 IDs de treino:")
    print(decoded_slice)


if __name__ == "__main__":
    main()
