"""Forward evaluation from previously fitted model coefficients.

Inference needs only the input features and saved coefficients. It does not
read the manuscript's target metrics or the training labels.
"""
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel


def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -40., 40.)))


def ann_forward(x, weights, h1, h2, mean, scale, ymean, yscale):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('Network requires (voltage_mV, age_years, BMI).')
    xs = (x - mean) / scale
    j, k = 3 * h1, 3 * h1 + h1 * h2
    w1 = weights[:j].reshape(3, h1)
    w2 = weights[j:k].reshape(h1, h2)
    w3 = weights[k:].reshape(h2)
    out = sigmoid(sigmoid(xs @ w1) @ w2) @ w3
    return out * float(np.asarray(yscale).item()) + float(np.asarray(ymean).item())


class FrozenPSOANN:
    """Serializable fixed-weight PSO–ANN inference, with no fitting methods."""
    def __init__(self, hidden, weights, xmean, xscale, ymean, yscale):
        self.h1, self.h2 = (int(hidden[0]), int(hidden[1]))
        self.weights = np.asarray(weights, dtype=float)
        self.xmean = np.asarray(xmean, dtype=float)
        self.xscale = np.asarray(xscale, dtype=float)
        self.ymean = float(ymean)
        self.yscale = float(yscale)

    def predict(self, x):
        return ann_forward(x, self.weights, self.h1, self.h2,
                           self.xmean, self.xscale, self.ymean, self.yscale)


class ArchitectureModel:
    def __init__(self, path):
        self.path = Path(path)
        self.d = np.load(self.path, allow_pickle=False)

    def predict(self, x):
        d = self.d
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        baseline = ann_forward(x, d['nn_weights'], int(d['h1']), int(d['h2']),
                               d['x_mean'], d['x_scale'], d['y_mean'], d['y_scale'])
        xs = (x - d['x_mean']) / d['x_scale']
        return baseline + rbf_kernel(xs, d['anchors_scaled'], gamma=float(d['gamma'])) @ d['kernel_dual']

    def predict_base(self, x):
        d = self.d
        return ann_forward(x, d['nn_weights'], int(d['h1']), int(d['h2']),
                           d['x_mean'], d['x_scale'], d['y_mean'], d['y_scale'])


class RegressionModel:
    def __init__(self, path):
        self.d = joblib.load(path)

    def predict(self, x):
        """Regression correction requires voltage, age, BMI, and device glucose."""
        d = self.d
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError('Regression calibration requires four features, including device_mgdl.')
        z = (x - d['correction_xmean']) / d['correction_xscale']
        distance = cdist(z, d['anchors_scaled'], 'sqeuclidean')
        kernel = np.exp(-distance / d['kernel_bandwidth_sq'])
        return self.base_predict(x[:, :3]) + kernel @ d['dual']

    def base_predict(self, x):
        d = self.d
        xs = d['base_scaler'].transform(x)
        if d['model_name'] == 'Polynomial Regression':
            xs = d['poly'].transform(xs)
        if d['model_name'] == 'PSO-ANN':
            return d['model'].predict(x)
        return d['model'].predict(xs)
