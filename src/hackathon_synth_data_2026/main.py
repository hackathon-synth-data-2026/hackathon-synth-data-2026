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

"""Run utility and fidelity evaluation using one configuration."""

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
        description="Evaluate synthetic-data utility and fidelity."
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

    # Read the JSON reports rather than serializing fidelity's
    # in-memory return value, which also contains DataFrames.
    metrics_dir = (
        Path(config["evaluation_output_path"]) / "metrics"
    )
    report = {}

    for name in ("utility", "fidelity"):
        report_file = metrics_dir / f"{name}.json"

        with report_file.open(encoding="utf-8") as handle:
            report[name] = json.load(handle)

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

    print(f"Combined report: {report_path}")


if __name__ == "__main__":
    main()