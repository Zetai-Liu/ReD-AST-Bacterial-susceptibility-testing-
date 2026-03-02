from pathlib import Path

from red_ast.pipeline import TARGET, load_dataset, train_and_evaluate


def test_load_dataset_columns_exist() -> None:
