import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.synth import GENERATORS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, generator in GENERATORS.items():
        df = generator()
        df.to_csv(DATA_DIR / filename, index=False)
        print(f"wrote {DATA_DIR / filename} ({len(df)} rows)")


if __name__ == "__main__":
    main()
