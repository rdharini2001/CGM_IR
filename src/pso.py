"""Particle swarm optimization of a small feedforward glucose regressor."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class PSOANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(3, 2), n_particles=25, n_iter=70, inertia=1.25,
                 cognitive=2.76, social=2.34, seed=2023, weight_bound=5.0,
                 velocity_bound=0.75):
        self.hidden = hidden
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.seed = seed
        self.weight_bound = weight_bound
        self.velocity_bound = velocity_bound

    def _number_weights(self):
        h1, h2 = self.hidden
        return 3*h1 + h1*h2 + h2

    def _forward(self, swarm, X):
        h1, h2 = self.hidden
        n = swarm.shape[0]
        idx1 = 3*h1
        idx2 = idx1 + h1*h2
        w1 = swarm[:, :idx1].reshape(n, 3, h1)
        w2 = swarm[:, idx1:idx2].reshape(n, h1, h2)
        w3 = swarm[:, idx2:].reshape(n, h2)
        a = sigmoid(np.einsum('sf,pfh->psh', X, w1))
        b = sigmoid(np.einsum('psh,phk->psk', a, w2))
        return np.einsum('psk,pk->ps', b, w3)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or len(X) != len(y):
            raise ValueError('Expected three inputs (voltage, age, BMI) and one reference glucose per sample')
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        Xs = self.x_scaler_.transform(X)
        ys = self.y_scaler_.transform(y.reshape(-1, 1)).ravel()
        rng = np.random.default_rng(self.seed)
        particles = rng.uniform(-1.0, 1.0, (self.n_particles, self._number_weights()))
        velocity = rng.uniform(-0.10, 0.10, particles.shape)
        personal_best = particles.copy()
        personal_losses = np.mean((self._forward(particles, Xs) - ys[None, :])**2, axis=1)
        idx = int(np.argmin(personal_losses))
        global_best = personal_best[idx].copy()
        global_loss = float(personal_losses[idx])
        self.history_ = [global_loss]
        for _ in range(self.n_iter):
            r1 = rng.random(particles.shape)
            r2 = rng.random(particles.shape)
            velocity = np.clip(
                self.inertia*velocity + self.cognitive*r1*(personal_best-particles)
                + self.social*r2*(global_best[None, :]-particles),
                -self.velocity_bound, self.velocity_bound)
            particles = np.clip(particles+velocity, -self.weight_bound, self.weight_bound)
            losses = np.mean((self._forward(particles, Xs)-ys[None, :])**2, axis=1)
            improved = losses < personal_losses
            personal_best[improved] = particles[improved]
            personal_losses[improved] = losses[improved]
            idx = int(np.argmin(personal_losses))
            if personal_losses[idx] < global_loss:
                global_best = personal_best[idx].copy()
                global_loss = float(personal_losses[idx])
            self.history_.append(global_loss)
        self.weights_ = global_best
        return self

    def predict(self, X):
        Xs = self.x_scaler_.transform(np.asarray(X, float))
        ys = self._forward(self.weights_[None, :], Xs)[0]
        return self.y_scaler_.inverse_transform(ys.reshape(-1, 1)).ravel()
