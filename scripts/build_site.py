"""Build a static project page from the saved forward model evaluations."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

def as_rows(path: Path):
    return json.loads(pd.read_csv(path).to_json(orient="records", double_precision=12))

def main():
    scores = pd.read_csv(ROOT / "results/forward_only_scores.csv")
    predictions = as_rows(ROOT / "results/forward_only_predictions.csv")

    def score(family, model, metric):
        subset = scores[(scores.family == family) & (scores.model == model) & (scores.metric == metric)]
        if len(subset) != 1:
            raise ValueError(f"Expected one score: {family}, {model}, {metric}")
        return float(subset.iloc[0].value)

    # The website is built entirely from saved-model scores and predictions.
    # No manuscript-target tables are needed for display or forward evaluation.
    architecture_names = list(scores[(scores.family == "architecture") &
                                     (scores.metric == "rmse")].sort_values("value", ascending=False).model)
    regression_names = list(scores[(scores.family == "regression") &
                                   (scores.metric == "mae")].model)
    architecture = [
        {"architecture": name,
         "validation_rmse": score("architecture", name, "rmse"),
         "accuracy_percent": score("architecture", name, "accuracy_weighted_r2_percent")}
        for name in architecture_names
    ]
    regression = [
        {"model": name, **{metric: score("regression", name, metric)
                              for metric in ("mae", "mse", "rmse", "r2", "modified_r2")}}
        for name in regression_names
    ]
    neural_weights = {}
    for item in architecture:
        name = item['architecture']
        with np.load(ROOT / 'models' / f'architecture_{name}.npz', allow_pickle=False) as fitted:
            neural_weights[name] = {
                'weights': fitted['nn_weights'].astype(float).tolist(),
                'kernel_alpha': float(fitted['kernel_alpha']),
                'gamma': float(fitted['gamma']),
                'scoring_effective_n': float(1.0 / np.sum(fitted['score_weights'] ** 2)),
                'x_mean': fitted['x_mean'].tolist(),
                'x_scale': fitted['x_scale'].tolist(),
                'y_mean': float(fitted['y_mean'].item()),
                'y_scale': float(fitted['y_scale'].item()),
                'gamma': float(fitted['gamma']),
                'kernel_dual': fitted['kernel_dual'].tolist(),
                'anchors_scaled': fitted['anchors_scaled'].tolist(),
            }
    curves = pd.read_csv(ROOT / 'results/optimizer_learning_histories.csv')
    curves = curves[curves.optimizer.isin(['PSO', 'Adam_target_inverse'])]
    histories = {name: [{'iteration': int(r.iteration), 'loss': float(r.normalized_mse)}
                        for r in curves[curves.optimizer == name].sort_values('iteration').itertuples(index=False)]
                 for name in ('PSO', 'Adam_target_inverse')}
    assert all(len(h) == 70 for h in histories.values())
    recipe_summary_path=ROOT / 'results' / 'weight_recipe_summary.json'
    recipe_summary = json.loads(recipe_summary_path.read_text()) if recipe_summary_path.exists() else {}
    trace_path=ROOT / 'results' / 'forward_trace_example.json'
    example_trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    recipe_seed_path=ROOT / 'results' / 'seed_sensitivity.csv'
    recipe_seed_rows = as_rows(recipe_seed_path) if recipe_seed_path.exists() else []
    study = {
        "training_recipe": recipe_summary,
        "training_seed_sensitivity": recipe_seed_rows,
        "forward_trace_example": example_trace,
        "source": "Published measurements; saved calibrated models",
        "doi": "10.36227/techrxiv.24465955.v1",
        "architectures": architecture,
        "regressions": regression,
        "networks": neural_weights,
        "learning_histories": histories,
        "observed": as_rows(ROOT / "data/observed_30.csv"),
        "hba": as_rows(ROOT / "data/hba1c.csv"),
        "predictions": predictions,
        "provenance": {
            "architecture_fit_target_conditioned": True,
            "regression_fit_target_conditioned": True,
            "observed_glucose_pairs": 30,
            "model_predictions": len(predictions),
            "clarke_scenario_has_simulated_rows": True,
        },
    }
    assert len(architecture) == 8 and len(regression) == 7
    assert len(study["observed"]) == 30 and len(study["hba"]) == 10
    assert len(predictions) == 450
    for r in regression:
        assert all(pd.notna(r[k]) for k in ("mae", "mse", "rmse", "r2", "modified_r2"))
    json_blob = json.dumps(study, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    json_blob = json_blob.replace("<", "\\u003c")
    template = (DOCS / "site.template.html").read_text(encoding="utf-8")
    if template.count("__STUDY_DATA__") != 1:
        raise ValueError("Missing or duplicate data placeholder")
    page = template.replace("__STUDY_DATA__", json_blob)
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print(f"Built docs/index.html | {len(architecture)} ANN models, {len(regression)} regressors, "
          f"{len(predictions)} fitted predictions, 30 observed glucose comparisons")
    return study

if __name__ == "__main__":
    main()
