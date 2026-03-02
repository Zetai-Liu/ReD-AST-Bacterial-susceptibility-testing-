#!/usr/bin/env python3
from __future__ import annotations

import argparse

from red_ast.pipeline import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline ReD-AST classifier")
    parser.add_argument(
        "--data",
        default="data/sample/reduced_ast_sample.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument("--model-out", default="reports/model.json")
    parser.add_argument("--report-out", default="reports/metrics.txt")
    args = parser.parse_args()

    result = train_and_evaluate(args.data, args.model_out, args.report_out)
    print("Training complete")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Model saved to: {result.model_path}")
    print(f"Report saved to: {result.report_path}")


if __name__ == "__main__":
    main()
