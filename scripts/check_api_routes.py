"""Dump every route registered on the FastAPI app for README cross-checking."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app


def main() -> None:
    schema = app.openapi()
    for path in sorted(schema["paths"]):
        for method in sorted(schema["paths"][path]):
            print(f"{method.upper():8s} {path}")


if __name__ == "__main__":
    main()
