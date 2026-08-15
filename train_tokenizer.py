"""
Treina um tokenizador BPE com os arquivos .txt válidos em data/
e salva em data/bpe_vocab.json.
"""
import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

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
    parser = argparse.ArgumentParser(description="Treina um tokenizador BPE em data/*.txt")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DATA_DIR, "bpe_vocab.json"),
        help="Caminho do tokenizador treinado",
    )
    args = parser.parse_args()

    txt_files = list_valid_txt_files(args.data_dir)
    if not txt_files:
        print(f"ERRO: nenhum .txt válido encontrado em {args.data_dir}")
        return

    print(f"Arquivos usados ({len(txt_files)}):")
    for p in txt_files:
        print(f"  - {os.path.basename(p)}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )

    tokenizer.train(files=txt_files, trainer=trainer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)

    trained_vocab_size = tokenizer.get_vocab_size()
    print(f"\nTokenizador salvo em: {args.output}")
    print(f"vocab_size solicitado: {args.vocab_size}")
    print(f"vocab_size treinado:   {trained_vocab_size}")


if __name__ == "__main__":
    main()
