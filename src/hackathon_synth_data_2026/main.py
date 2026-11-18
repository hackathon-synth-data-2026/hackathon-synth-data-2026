# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0,<3",
#     "pandas==2.3.3",
#     "pyarrow>=24.0.0",
#     "matplotlib>=3.10.9",
#     "scikit-learn==1.9.1",
#     "scipy>=1.12.1",
#     "xgboost==3.4.1",
# ]
# ///

"""
Evaluate synthetic-data utility, fidelity, and privacy.

Expected files
--------------
    evaluation/
    ├── main.py
    ├── utility.py
    ├── fidelity.py
    ├── privacy.py
    └── config.json

Input data
----------
Real training, real test, and synthetic datasets must contain complete
feature-and-target tables. Do not supply X and y as separate files.

The target column is specified by data_specs.target_var and must contain
nonmissing binary labels: 0 = survival, 1 = in-hospital death.

Dataset paths, feature definitions, and evaluation settings are supplied
through config.json. CSV and Parquet are supported by the evaluators.

Usage examples
--------------
Run these commands from the starterpack root.

1. Use config.json beside main.py, including its synthetic_data_path:

    uv run evaluation/main.py

2. Specify a configuration file explicitly:

    uv run evaluation/main.py evaluation/config.json

3. Evaluate your own synthetic dataset:

    uv run evaluation/main.py --synth data/submission.parquet

4. Specify both the configuration and synthetic dataset:

    uv run evaluation/main.py evaluation/config.json --synth data/submission.csv

5. Save reports in a different output directory:

    uv run evaluation/main.py --synth data/submission.parquet --output-path results/my-run

6. Run only the utility evaluator's submission-schema checks:

    uv run evaluation/main.py --synth data/submission.parquet --validate-only

    This skips all metric calculations. It does not check every
    fidelity- or privacy-specific requirement.

7. Show command-line help:

    uv run evaluation/main.py --help

Path resolution
---------------
- The default configuration is config.json beside main.py.
- Relative paths inside config.json are resolved relative to that file.
- Paths passed through --synth and --output-path are resolved relative
  to your current working directory.
- CLI overrides take precedence over the corresponding config values.
- validate_only=true in config.json also skips metric calculations.

Outputs
-------
Reports are saved under evaluation_output_path:

    utility/metrics/utility.json
    fidelity/metrics/fidelity.json
    privacy/metrics/privacy.json
    evaluation.json

Individual evaluators also produce plots, CSV diagnostics, and artifacts.
evaluation.json combines the three reports; it does not calculate a
single overall leaderboard score.

Reproducible environment
------------------------
Create a script lockfile and distribute it with the starterpack:

    uv lock --script evaluation/main.py

Then run with the locked dependencies:

    uv run --locked evaluation/main.py --synth data/submission.parquet
"""

import argparse
import json
from pathlib import Path


def load_config(config_path):
    """Resolve configuration paths relative to the configuration file."""
    config_path = Path(config_path).resolve()

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    for key in (
        "real_train_path",
        "real_test_path",
        "real_data_path",
        "synthetic_data_path",
        "evaluation_output_path",
    ):
        if config.get(key):
            path = Path(config[key]).expanduser()

            if not path.is_absolute():
                path = config_path.parent / path

            config[key] = str(path.resolve())

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic-data utility, fidelity, and privacy."
    )
    parser.add_argument(
        "experiment_config",
        nargs="?",
        default=str(Path(__file__).with_name("config.json")),
        help="Config JSON; defaults to config.json beside main.py",
    )
    parser.add_argument(
        "--synth",
        help="Override synthetic_data_path",
    )
    parser.add_argument(
        "--output-path",
        help="Override evaluation_output_path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run the utility evaluator's schema checks "
            "and skip metric evaluation"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.experiment_config)

    # CLI paths follow normal command-line conventions:
    # they are relative to the current working directory.
    if args.synth:
        config["synthetic_data_path"] = str(
            Path(args.synth).expanduser().resolve()
        )

    if args.output_path:
        config["evaluation_output_path"] = str(
            Path(args.output_path).expanduser().resolve()
        )

    if args.validate_only:
        config["validate_only"] = True

    if not config.get("synthetic_data_path"):
        parser.error(
            "Pass --synth or set synthetic_data_path in config.json"
        )

    if not config.get("evaluation_output_path"):
        parser.error(
            "Set evaluation_output_path or pass --output-path"
        )

    if (
        "data_specs" not in config
        or "target_var" not in config["data_specs"]
    ):
        parser.error("Set data_specs.target_var in config.json")

    from utility import run_utility_eval
    synthetic_path = config["synthetic_data_path"]
    print("\n=== Utility evaluation ===", flush=True)
    run_utility_eval(config, synthetic_path)
    if config.get("validate_only", False):
        print(
            "\nSchema validation complete; "
            "metric evaluation skipped."
        )
        return

    from fidelity import run_fidelity_eval
    print("\n=== Fidelity evaluation ===", flush=True)
    run_fidelity_eval(config, synthetic_path)
    from privacy import run_privacy_eval
    print("\n=== Privacy evaluation ===", flush=True)
    run_privacy_eval(config, synthetic_path)

    # Read the JSON reports rather than serializing fidelity's
    # in-memory return value, which also contains DataFrames.
    metrics_dir = (
        Path(config["evaluation_output_path"])
    )
    report = {}
    for name in ("utility", "fidelity", "privacy"):
        report_file = metrics_dir / name / "metrics" / f"{name}.json"
        with report_file.open(encoding="utf-8") as handle:
            report[name] = json.load(handle)

    from scoring import calculate_final_score
    report["ranking"] = calculate_final_score(
    report,
    weights={
        "utility": 1/3,
        "fidelity": 1/3,
        "privacy": 1/3})
    report_path = metrics_dir / "evaluation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(
        "Final ranking points: "
        f"{report['ranking']['final_points']:.2f}"
    )

    report_path = metrics_dir / "evaluation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Evaluation complete ===")
    print(
        "TSTR AUROC: "
        f"{report['utility']['tstr']['auc']:.3f}"
    )
    print(
        "TSTR average precision: "
        f"{report['utility']['tstr']['pr_auc']:.3f}"
    )

    score = report["fidelity"]["score"]
    if score is None:
        print("Fidelity score: undefined; inspect fidelity.json")
    else:
        print(f"Fidelity score: {score:.3f}")

    privacy_summary = report["privacy"]["summary"]
    print(
        "Privacy score: "
        f"{privacy_summary['privacy_score']:.3f}")
    print(
        "Privacy score basis: "
        f"{privacy_summary['score_basis']}")

    print(f"Combined report: {report_path}")


if __name__ == "__main__":
    main()