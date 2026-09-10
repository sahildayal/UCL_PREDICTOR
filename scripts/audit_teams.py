"""Scan the club registry for near-duplicate names.

A club split across two spellings gets half its match history each, and the half
attached to the current-season fixture list has almost none - so it inherits its
league's average and is rated as though it were a typical club from that country.
That is how Lens ended up rated on France's league prior instead of its own
results, and it is invisible unless something looks for it.

Run this after any corpus rebuild. Pairs that are genuinely different clubs
belong in teams.DO_NOT_MERGE, not in the alias table.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl.config import force_utf8  # noqa: E402
from ucl.teams import DO_NOT_MERGE, TeamRegistry  # noqa: E402

force_utf8()


def main() -> int:
    registry = TeamRegistry.load()
    if not len(registry):
        print("No registry found. Build the corpus first.")
        return 1

    keys = registry.keys()
    seen: set[tuple[str, str]] = set()
    flagged: list[tuple[str, str, str, str]] = []

    for key in keys:
        for other in difflib.get_close_matches(key, keys, n=4, cutoff=0.86):
            if other == key:
                continue
            pair = tuple(sorted((key, other)))
            if pair in seen or pair in DO_NOT_MERGE:
                continue
            seen.add(pair)
            flagged.append((pair[0], pair[1],
                            registry.country_of(pair[0]),
                            registry.country_of(pair[1])))

    print(f"clubs in registry: {len(registry)}")
    print(f"unreviewed near-duplicate pairs: {len(flagged)}\n")
    for a, b, ca, cb in sorted(flagged):
        marker = "  <-- same country" if ca and ca == cb else ""
        print(f"  {a:<32} ({ca or '?'})   ~   {b:<32} ({cb or '?'}){marker}")

    if flagged:
        print("\nAdd genuine duplicates to teams._ALIASES; add genuine distinct "
              "clubs to teams.DO_NOT_MERGE.")
    return 2 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
