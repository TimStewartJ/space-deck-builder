"""Check the held-out league port against the 2026-09-30 prototype counts.

Decision 0013 requires the ported league to reproduce the prototype's wins
for the published selfplay_seed0 checkpoint, and each held-out opponent's
wins against Heuristic, before league numbers are reported.

Usage: python results/strength_yardstick/check_port.py <league.json> [<report.json>]
"""
from __future__ import annotations

import json
import sys

CHAMPION_SHA256 = "9d10dca5110c2bbe9cc0f8693e11578dee347a8c60e7ed63002af7d82e80a503"
CHAMPION_WINS = {
    "balanced": 3084, "aggro": 3321, "economy": 3111,
    "scrapper": 3055, "fortress": 3092, "blobrush": 3345,
}
CALIBRATION_WINS = {
    "balanced": 1683, "aggro": 1533, "economy": 1589,
    "scrapper": 1664, "fortress": 1670, "blobrush": 1479,
}


def check(league: dict) -> dict:
    champion = next(
        (row for row in league["checkpoints"] if row["sha256"] == CHAMPION_SHA256), None,
    )
    rows = []
    for opponent, expected in CHAMPION_WINS.items():
        actual = None if champion is None else champion["opponents"].get(opponent, {}).get("wins")
        rows.append({"table": "champion", "opponent": opponent, "expected": expected, "actual": actual})
    for opponent, expected in CALIBRATION_WINS.items():
        actual = league.get("calibration", {}).get(opponent, {}).get("wins")
        rows.append({"table": "calibration", "opponent": opponent, "expected": expected, "actual": actual})
    return {
        "passed": all(row["actual"] == row["expected"] for row in rows),
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as fh:
        report = check(json.load(fh))
    for row in report["rows"]:
        mark = "ok" if row["actual"] == row["expected"] else "MISMATCH"
        print(f"{row['table']:>11} {row['opponent']:<9} expected {row['expected']} actual {row['actual']}  {mark}")
    print("port check passed" if report["passed"] else "port check FAILED")
    if len(argv) > 2:
        with open(argv[2], "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
