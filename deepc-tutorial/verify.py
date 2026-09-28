# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "marimo==0.25.0",
#     "numpy==2.5.3",
#     "cvxpy==1.9.3",
#     "matplotlib==3.11.2",
#     "osqp==1.1.3",
#     "clarabel>=0.11,<0.12",
# ]
# ///
"""Execute the actual notebook, check edge settings, optionally save results."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from deepc import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save default-run plots and validation JSON")
    args = parser.parse_args()
    checks = {}
    for weight in (0.1, 0.01, 1.0):
        _, definitions = app.run(defs={"effort": SimpleNamespace(value=weight)})
        checks[str(weight)] = dict(definitions["verification"])
        np.testing.assert_array_equal(
            definitions["hankel"]([1, 2, 3, 4, 5], 3),
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        )
        # Check time-major stacking for two channels, not only the SISO example.
        np.testing.assert_array_equal(
            definitions["hankel"]([[1, 10], [2, 20], [3, 30]], 2),
            [[1, 2], [10, 20], [2, 3], [20, 30]],
        )
        assert np.linalg.matrix_rank(definitions["hankel"](np.ones(160), 18)) == 1
        noisy = definitions["noisy_run"]
        assert all(np.isfinite(values).all() for values in noisy.values())
        assert np.max(np.abs(noisy["u"])) <= 1.0 + 1e-6
        if args.output and weight == 0.1:
            args.output.mkdir(parents=True, exist_ok=True)
            for key in ("data_figure", "tracking_figure", "noise_figure"):
                definitions[key].savefig(args.output / f"{key}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    report = {"status": "passed", "effort_weights": checks,
              "additional_checks": ["Hankel indexing", "multi-channel stacking",
                                    "constant input fails excitation", "noisy solve and input bounds"]}
    if args.output:
        (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
