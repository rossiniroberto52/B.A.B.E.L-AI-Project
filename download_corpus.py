"""
Baixa uma amostra de texto em português (CC100) para pré-treino.
Salva em data/cc100_ptbr.txt
"""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TARGET_SIZE_MB = 600
TARGET_BYTES = TARGET_SIZE_MB * 1024 * 1024


def main():
    from datasets import load_dataset

    out_path = os.path.join(DATA_DIR, "cc100_ptbr.txt")
    print(f"Baixando CC100 PT-BR (streaming) -> {out_path}")
    print(f"Meta: ~{TARGET_SIZE_MB} MB de texto bruto")

    ds = load_dataset("cc100", "pt", split="train", streaming=True)

    written = 0
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue
            f.write(text + "\n\n")
            written += len(text.encode("utf-8"))
            count += 1
            if count % 1000 == 0:
                mb = written / (1024 * 1024)
                print(f"  {count} docs | {mb:.1f} MB", flush=True)
            if written >= TARGET_BYTES:
                break

    final_mb = written / (1024 * 1024)
    print(f"\nPronto! {count} docs | {final_mb:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
