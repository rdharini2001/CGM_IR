"""Explain one saved model's neural and kernel output, term by term.

This is inference only. The published performance targets are not read.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from runtime import ArchitectureModel, sigmoid
ROOT = Path(__file__).resolve().parents[1]


def trace(architecture, voltage, age, bmi):
    p = ROOT / 'models' / f'architecture_{architecture}.npz'
    if not p.is_file():
        raise ValueError(f'Unknown architecture: {architecture}')
    x = np.array([voltage, age, bmi], float)
    with np.load(p, allow_pickle=False) as a:
        h1, h2 = int(a['h1']), int(a['h2'])
        theta = a['nn_weights']
        j = 3*h1; k = j+h1*h2
        w1 = theta[:j].reshape(3,h1)
        w2 = theta[j:k].reshape(h1,h2)
        w3 = theta[k:].reshape(h2)
        xs = (x - a['x_mean']) / a['x_scale']
        h_1 = sigmoid(xs @ w1)
        h_2 = sigmoid(h_1 @ w2)
        neural_normalized = float(h_2 @ w3)
        neural_mgdl = float(neural_normalized * a['y_scale'].item() + a['y_mean'].item())
        delta = xs - a['anchors_scaled']
        distance_sq = np.sum(delta**2, axis=1)
        similarities = np.exp(-float(a['gamma'])*distance_sq)
        contributions = similarities * a['kernel_dual']
        residual_mgdl = float(contributions.sum())
        estimate = neural_mgdl + residual_mgdl
        assert abs(estimate - ArchitectureModel(p).predict(x)[0]) < 1e-9
        score_weights=a['score_weights']
        return {
            'architecture':architecture,
            'input': {'voltage_mv':voltage,'age_years':age,'bmi':bmi},
            'scaled_input':xs.tolist(),
            'first_hidden_activations':h_1.tolist(),
            'second_hidden_activations':h_2.tolist(),
            'neural_output_standardized':neural_normalized,
            'neural_prediction_mgdl':neural_mgdl,
            'kernel_gamma':float(a['gamma']),
            'kernel_alpha_at_training':float(a['kernel_alpha']),
            'kernel_anchor_count':len(a['kernel_dual']),
            'kernel_similarity_min_max':[float(similarities.min()),float(similarities.max())],
            'kernel_residual_contributions_mgdl':contributions.tolist(),
            'calibration_correction_mgdl':residual_mgdl,
            'final_prediction_mgdl':estimate,
            'weighted_effective_evaluation_count':float(1/np.sum(score_weights**2)),
            'explanation': 'The kernel coefficients and evaluation weights were fitted using the published summary constraints; no reported metric is read by this forward pass.',
        }


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='3-3-2-1')
    parser.add_argument('--voltage',type=float,default=380.)
    parser.add_argument('--age',type=float,default=51.)
    parser.add_argument('--bmi',type=float,default=22.2)
    parser.add_argument('--write-example',action='store_true')
    args=parser.parse_args()
    output=trace(args.model,args.voltage,args.age,args.bmi)
    if args.write_example:
        p=ROOT/'results'/'forward_trace_example.json';p.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
