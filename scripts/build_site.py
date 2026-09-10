"""Regenerate the GitHub Pages site from the archived briefs.

Idempotent, and safe to run any time: it reads only committed briefs and current
results, so it reconstructs the same site from the same history.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from ucl.config import force_utf8  # noqa: E402
from ucl.data.results import results_map  # noqa: E402
from ucl.report.site import build_site  # noqa: E402

force_utf8()


def main() -> int:
    try:
        results = results_map()
    except Exception as exc:                # noqa: BLE001
        print(f"could not load results ({type(exc).__name__}: {exc}); "
              f"building site without them")
        results = {}
    out = build_site(results)
    pages = sorted(p.name for p in out.glob("*.html"))
    print(f"built {len(pages)} pages in {out}")
    for name in pages:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
