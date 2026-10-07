"""Independent checks on saved evidence; does not train or score the test set."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    result = ROOT / "results"
    source = ROOT / "data" / "raw" / "bank-additional-full.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_evaluated"] is False
    manifest = pd.read_csv(result / "split_manifest.csv", dtype={"predictor_group": str})
    assert len(manifest) == 41188 and manifest["row_id"].nunique() == 41188
    assert set(manifest["row_id"]) == set(range(41188))
    assert manifest.groupby("predictor_group")["split"].nunique().max() == 1
    folds = pd.read_csv(result / "cv_fold_manifest.csv", dtype={"predictor_group": str})
    train_ids = set(manifest.loc[manifest["split"] == "train", "row_id"])
    val_ids = set(manifest.loc[manifest["split"] == "validation", "row_id"])
    assert set(folds["row_id"]) == train_ids and folds["row_id"].is_unique
    assert set(folds["cv_fold"]) == set(range(5))
    assert folds.groupby("predictor_group")["cv_fold"].nunique().max() == 1
    predictions = pd.read_csv(result / "validation_predictions.csv")
    metrics = pd.read_csv(result / "metrics.csv")
    raw = pd.read_csv(source, sep=";")
    assert set(predictions["model"]) == {"Dummy (majority)", "Decision Tree", "KNN", "Linear SVM"}
    functions = {"f1_yes": f1_score, "precision_yes": precision_score,
                 "recall_yes": recall_score, "roc_auc": roc_auc_score,
                 "average_precision": average_precision_score}
    for model, values in predictions.groupby("model"):
        assert set(values["row_id"]) == val_ids and values["row_id"].is_unique
        expected_y = raw.loc[values["row_id"], "y"].map({"no": 0, "yes": 1}).to_numpy()
        np.testing.assert_array_equal(expected_y, values["y_true"].to_numpy())
        stored = metrics[(metrics["model"] == model) & (metrics["split"] == "validation")].iloc[0]
        for name, function in functions.items():
            if name in {"roc_auc", "average_precision"}:
                actual = function(values["y_true"], values["score"])
            else:
                actual = function(values["y_true"], values["y_pred"], zero_division=0)
            assert np.isclose(actual, stored[name], atol=1e-12), (model, name)
    baseline = metrics[(metrics["model"] == "Dummy (majority)") & (metrics["split"] == "validation")].iloc[0]
    assert baseline["f1_yes"] == baseline["recall_yes"] == 0
    notebook = nbformat.read(ROOT / "notebooks" / "bank_marketing.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    source_sections = re.split(r"^# %%", (ROOT / "src" / "bank_marketing.py").read_text(encoding="utf-8"), flags=re.MULTILINE)
    source_code = [section.strip() for section in source_sections if section.strip() and not section.startswith(" [markdown]")]
    assert [cell.source for cell in code_cells] == source_code, "Notebook/source code differs; synchronize and rerun."
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    assert len(list((result / "figures").glob("*.png"))) >= 9
    checks = {"status": "passed", "source_checksum": True, "complete_disjoint_split": True,
              "group_disjoint_cv": True, "test_not_scored": True,
              "saved_metrics_independently_recomputed": True,
              "notebook_matches_python_code": True,
              "notebook_executed_without_errors": True, "code_cells": len(code_cells)}
    (result / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__":
    main()
