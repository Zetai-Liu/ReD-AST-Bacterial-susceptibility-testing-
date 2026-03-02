from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

NUMERIC_FEATURES = ["sensor_peak", "sensor_auc", "incubation_min"]
CATEGORICAL_FEATURES = ["strain_group"]
TARGET = "label"


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    model_path: Path
    report_path: Path


def load_dataset(csv_path: str | Path) -> List[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required_cols - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for row in rows:
        for col in NUMERIC_FEATURES:
            row[col] = float(row[col])
        row[TARGET] = int(row[TARGET])
    return rows


def split_xy(rows: List[dict]) -> Tuple[List[dict], List[int]]:
    x = [{k: v for k, v in r.items() if k != TARGET} for r in rows]
    y = [r[TARGET] for r in rows]
    return x, y


def train_simple_auc_threshold(x: List[dict], y: List[int]) -> dict:
    pos = [row["sensor_auc"] for row, label in zip(x, y) if label == 1]
    neg = [row["sensor_auc"] for row, label in zip(x, y) if label == 0]
    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)
    threshold = (pos_mean + neg_mean) / 2.0
    return {
        "feature": "sensor_auc",
        "threshold": threshold,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def predict(model: dict, x: List[dict]) -> List[int]:
    return [1 if row[model["feature"]] >= model["threshold"] else 0 for row in x]


def predict_proba(model: dict, x: List[dict]) -> List[float]:
    scale = max(abs(model["pos_mean"] - model["neg_mean"]), 1e-6)
    probs = []
    for row in x:
        z = (row[model["feature"]] - model["threshold"]) / scale
        probs.append(1.0 / (1.0 + math.exp(-z)))
    return probs


def train_test_split(x: List[dict], y: List[int], test_ratio: float = 0.25) -> Tuple[List[dict], List[dict], List[int], List[int]]:
    n = len(x)
    test_n = max(1, int(n * test_ratio))
    train_n = n - test_n
    return x[:train_n], x[train_n:], y[:train_n], y[train_n:]


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def roc_auc_score(y_true: List[int], y_score: List[float]) -> float:
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / total


def train_and_evaluate(
    csv_path: str | Path,
    model_output: str | Path = "reports/model.json",
    report_output: str | Path = "reports/metrics.txt",
) -> TrainResult:
    rows = load_dataset(csv_path)
    x, y = split_xy(rows)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25)

    model = train_simple_auc_threshold(x_train, y_train)
    preds = predict(model, x_test)
    probs = predict_proba(model, x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ReD-AST baseline metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return TrainResult(metrics=metrics, model_path=model_path, report_path=report_path)
