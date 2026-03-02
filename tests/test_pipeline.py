from pathlib import Path

from red_ast.pipeline import TARGET, load_dataset, train_and_evaluate


def test_load_dataset_columns_exist() -> None:
    rows = load_dataset("data/sample/reduced_ast_sample.csv")
    assert TARGET in rows[0]
    assert len(rows) >= 10


def test_train_and_evaluate_outputs(tmp_path: Path) -> None:
    model_out = tmp_path / "model.json"
    report_out = tmp_path / "metrics.txt"

    result = train_and_evaluate(
        "data/sample/reduced_ast_sample.csv",
        model_output=model_out,
        report_output=report_out,
    )

    assert model_out.exists()
    assert report_out.exists()
    assert set(result.metrics.keys()) == {"accuracy", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in result.metrics.values())
