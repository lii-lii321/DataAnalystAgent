"""Probe: /healthz responsiveness while a blocking /analyze request is in flight.

Passes when /healthz (fired ~10 ms into analyze) returns in milliseconds
instead of being blocked until the analysis finishes.
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent.config import settings  # noqa: E402
from agent.synth import make_income  # noqa: E402
from api.store import REGISTRY  # noqa: E402
from main import app  # noqa: E402


async def main() -> int:
    settings.artifacts_dir = tempfile.mkdtemp(prefix="daa_probe_artifacts_")
    REGISTRY.clear()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            df = make_income()
            upload = await client.post(
                "/datasets",
                files={"file": ("income.csv", df.to_csv(index=False).encode(), "text/csv")},
            )
            assert upload.status_code == 201, upload.status_code
            dataset_id = upload.json()["id"]

            t_start = time.perf_counter()
            analyze_task = asyncio.create_task(
                client.post(
                    f"/datasets/{dataset_id}/analyze",
                    json={"question": "男性和女性的 income 是否存在显著差异？"},
                )
            )
            await asyncio.sleep(0.01)
            t_health = time.perf_counter()
            health = await client.get("/healthz")
            health_ms = (time.perf_counter() - t_health) * 1000
            analyze = await analyze_task
            analyze_s = time.perf_counter() - t_start

            print(f"healthz : HTTP {health.status_code} in {health_ms:.1f} ms (fired +10 ms into analyze)")
            print(f"analyze : HTTP {analyze.status_code} in {analyze_s:.2f} s")
            ok = health.status_code == 200 and analyze.status_code == 200 and health_ms < 1000
            print("PASS" if ok else "FAIL")
            return 0 if ok else 1
    finally:
        REGISTRY.clear()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
