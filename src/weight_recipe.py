"""Deterministic neural-weight replay and seed-sensitivity experiment.

The replay target is the saved computational-release model, NOT original
unavailable weights from the authors' laboratory. The neural PSO objective
uses only the thirty published (voltage, age, BMI, glucose) observations;
no numerical table targets are loaded. The separate RBF correction is
aggregate-constrained and handled by train_realization.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pso import PSOANNRegressor
from runtime import ArchitectureModel

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('voltage_mv', 'age_years', 'bmi')
# The architecture ordering is part of the deterministic seed protocol.
SEED_OFFSETS = (0, 1, 7, 31, 113)


def train_network(architecture, seed, x, y):
    shape = tuple(map(int, architecture.split('-')))
    assert shape[0] == 3 and shape[-1] == 1
    return PSOANNRegressor(
        hidden=(shape[1], shape[2]),
        n_particles=25, n_iter=70, inertia=1.25,
        cognitive=2.76, social=2.34, seed=seed,
        weight_bound=5.0, velocity_bound=0.75,
    ).fit(x, y)


def execute(mode='all', strict=True):
    df = pd.read_csv(ROOT / 'data' / 'observed_30.csv')
    y = df.reference_mgdl.to_numpy(float)
    x = df[list(FEATURES)].to_numpy(float)
    names = pd.read_csv(ROOT / 'data' / 'paper_architectures.csv').architecture.to_list()
    (ROOT / 'results').mkdir(exist_ok=True)
    rows, paths, coefficients, traces = [], [], [], []
    for idx, name in enumerate(names):
        source_path = ROOT / 'models' / f'architecture_{name}.npz'
        with np.load(source_path, allow_pickle=False) as fitted:
            reference = fitted['nn_weights'].copy()
            reference_mean = fitted['x_mean'].copy()
            reference_scale = fitted['x_scale'].copy()
            reference_y_mean = fitted['y_mean'].copy()
            reference_y_scale = fitted['y_scale'].copy()
        base_predictions = ArchitectureModel(source_path).predict_base(x)
        offsets = SEED_OFFSETS if mode in ('sweep', 'all') else (0,)
        for offset in offsets:
            seed = 2026 + idx + offset
            net = train_network(name, seed, x, y)
            predicted = net.predict(x)
            difference = net.weights_ - reference
            matching = bool(np.array_equal(net.weights_, reference))
            score = float(np.sqrt(np.mean((predicted - y)**2)))
            row = dict(architecture=name, seed=seed, seed_offset=offset,
                       coefficients=len(net.weights_),
                       exact_weight_replay=matching,
                       max_abs_weight_error=float(np.max(np.abs(difference))),
                       weight_distance_l2=float(np.linalg.norm(difference)),
                       base_prediction_difference_rmse=float(np.sqrt(np.mean((predicted-base_predictions)**2))),
                       neural_fitted_rmse_mgdl=score,
                       final_normalized_training_mse=float(net.history_[-1]),
                       is_saved_recipe=(offset == 0))
            rows.append(row)
            if offset == 0:
                assert len(net.history_) == 71
                scaler_match = (np.array_equal(net.x_scaler_.mean_, reference_mean)
                                and np.array_equal(net.x_scaler_.scale_, reference_scale)
                                and np.array_equal(net.y_scaler_.mean_, reference_y_mean)
                                and np.array_equal(net.y_scaler_.scale_, reference_y_scale))
                if strict and not (np.allclose(net.weights_, reference, atol=1e-9, rtol=0) and
                                   np.allclose(net.x_scaler_.mean_, reference_mean, atol=1e-9, rtol=0) and
                                   np.allclose(net.x_scaler_.scale_, reference_scale, atol=1e-9, rtol=0) and
                                   np.allclose(net.y_scaler_.mean_, reference_y_mean, atol=1e-9, rtol=0) and
                                   np.allclose(net.y_scaler_.scale_, reference_y_scale, atol=1e-9, rtol=0)):
                    raise AssertionError(f'Network/scalers did not replay exactly for {name}')
                paths.append({'architecture': name, 'seed': seed,
                              'exact_neural_weight_replay': matching,
                              'exact_feature_and_label_scalers': scaler_match,
                              'neural_rmse_mgdl': score,
                              'max_abs_saved_weight_difference': row['max_abs_weight_error']})
                for j, w in enumerate(net.weights_):
                    coefficients.append({'architecture': name, 'weight_index': j,
                                         'weight_value': float(w),
                                         'seed': seed})
                for i, loss in enumerate(net.history_):
                    traces.append({'architecture': name, 'step': i, 'normalized_training_mse':float(loss)})
        print(f'{name}: replay={paths[-1]["exact_neural_weight_replay"]} '
              f'PSO seed={2026 + idx} base RMSE={paths[-1]["neural_rmse_mgdl"]:.3f} mg/dL')
    replay = pd.DataFrame(paths)
    sweep = pd.DataFrame(rows)
    replay.to_csv(ROOT / 'results' / 'weight_replay.csv', index=False)
    sweep.to_csv(ROOT / 'results' / 'seed_sensitivity.csv', index=False)
    pd.DataFrame(coefficients).to_csv(ROOT / 'results' / 'replayed_connection_weights.csv', index=False)
    pd.DataFrame(traces).to_csv(ROOT / 'results' / 'neural_pso_training_histories.csv', index=False)
    other = sweep[sweep.seed_offset.ne(0)]
    summary = {
        'protocol': '3-input no-bias sigmoid-sigmoid-linear; standardize X and y; 25 particles; 70 PSO steps',
        'data_source': '30 printed glucose observations from 10 subjects',
        'reference': 'saved OPTIC computational-release weights, NOT original clinical investigators weights',
        'training_reads_published_metric_targets': False,
        'replay_seed_rule': '2026 + zero-based architecture-table index',
        'architecture_count': len(names),
        'exact_weight_replays': int(replay.exact_neural_weight_replay.sum()),
        'exact_scaler_replays': int(replay.exact_feature_and_label_scalers.sum()),
        'seed_sensitivity_offsets': list(offsets),
        'alternative_seed_runs': int(len(other)),
        'alternative_seed_exact_weight_matches': int(other.exact_weight_replay.sum()),
        'alternative_seed_median_weight_distance_l2': float(other.weight_distance_l2.median()) if len(other) else None,
        'downstream_kernel_and_scorer_fit_use_manuscript_targets': True,
        'score_interpretation': 'Retrospective target-conditioned fit, not independent validation',
    }
    (ROOT / 'results' / 'weight_recipe_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Weight replay:',summary['exact_weight_replays'],'/',len(names),'| alternative seeds:',len(other))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('replay', 'sweep', 'all'), default='all')
    ap.add_argument('--no-strict', action='store_true', help='Record weight differences instead of rejecting a changed software stack')
    args = ap.parse_args()
    execute(mode=args.mode, strict=not args.no_strict)
