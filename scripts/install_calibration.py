"""Install a calibration table produced by a harness run.

    python -m evaluation.harness --input data/development_tickets.json --output evaluation/results/dev
    python -m scripts.install_calibration evaluation/results/dev

The classifier then corrects its stated confidence to the accuracy observed on the
development set. Calibrate on development data only: calibrating on the set you report
against makes the calibration figure meaningless.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.classify import CALIBRATION_FILE


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a calibration table.")
    parser.add_argument("run_dir", type=Path, help="A harness output directory")
    parser.add_argument("--force", action="store_true", help="Install even from a non-development run")
    args = parser.parse_args()

    source = args.run_dir / "calibration.json"
    if not source.exists():
        print(f"error: {source} not found")
        return 2

    report_file = args.run_dir / "metrics.json"
    if report_file.exists() and not args.force:
        input_file = json.loads(report_file.read_text(encoding="utf-8"))["run"]["input_file"]
        if "development" not in Path(input_file).name:
            print(
                f"refusing to calibrate on {Path(input_file).name}: calibrate on development "
                "data only, or pass --force"
            )
            return 2

    table = json.loads(source.read_text(encoding="utf-8"))
    usable = [b for b in table.get("bands", []) if b.get("count", 0) >= 10]
    if not usable:
        print("error: no confidence band has at least 10 predictions; nothing to install")
        return 2

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, CALIBRATION_FILE)
    print(f"installed {len(usable)} calibration band(s) from {source} -> {CALIBRATION_FILE}")
    for band in usable:
        print(
            f"  {band['low']:.1f}-{band['high']:.1f}  n={band['count']:<5} "
            f"stated {band['stated']:.3f} -> observed {band['observed']:.3f} "
            f"({band['gap_points']:+.1f} pts)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
