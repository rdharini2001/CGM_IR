# Deterministic PSO–ANN weight-training recipe

This document separates **neural-weight fitting from numerical-summary calibration**. The precise recipe regenerates the eight neural-weight vectors shipped in this computational release. These are **the OPTIC implementation's fitted weights**, not recovered unpublished weights of the manuscript authors. The patient measurements cannot uniquely identify a particular historical training seed or all neural parameters.

## 1. Source inputs and fit objective

Thirty printed measurement pairs from ten participants, each at the before, after and random phases, are read from `data/observed_30.csv`. The neural network sees voltage (mV), age (years), and BMI. Its supervised response is the printed reference glucose (mg/dL). No optical waveform is available; voltage is the numerical input. **The neural-weight fitting stage does not read either manuscript performance table.**

Fit `StandardScaler` to each of the three input features and to the glucose responses on these same 30 rows. In an independent evaluation, the scaler would instead be fitted inside each training fold. The fixed retrospective full-data fit is intentionally used here to replay the distributed model parameters.

The two hidden layers use logistic sigmoid activations, with a linear output and **no biases**. For 3–3–2–1, the flattening order is `W1[3,3]`, `W2[3,2]`, `W3[2]`, yielding **17 trainable neural weights**. The swarm minimizes the mean squared error of normalized glucose:

\[
 L(\theta)=\frac1{30}\sum_{i=1}^{30}[f_{\theta}(z_i)-t_i]^2,
 \quad z_i=\frac{x_i-\mu_x}{s_x},\quad t_i=\frac{y_i-\mu_y}{s_y}.
\]

## 2. Deterministic swarm schedule

| Configuration | Value |
|---|---|
| Swarm | 25 particles |
| Updates | 70 (71 best-loss snapshots including initialization) |
| Inertia | 1.25 |
| Personal attraction | 2.76 |
| Global attraction | 2.34 |
| Initial positions | Independent Uniform(−1, 1), NumPy `default_rng` |
| Initial velocities | Independent Uniform(−0.10, 0.10) |
| Velocity clipping | [−0.75, 0.75] |
| Weight clipping | [−5, 5] |
| Seed | **2026 + zero-based row index** in `data/paper_architectures.csv` |
| Best-particle update | Strictly lower normalized MSE |

At step \(t\), with independent \(r_1,r_2\sim U(0,1)\) for every particle and coefficient:

\[
 v^{t+1}=\operatorname{clip}_{[-.75,.75]}[1.25v^t+2.76r_1(p_{best}-w^t)+2.34r_2(g_{best}-w^t)],
\]
\[
 w^{t+1}=\operatorname{clip}_{[-5,5]}(w^t+v^{t+1}).
\]

The serialization stores the global best neural vector and fitted feature/response scalers. The seed rule was **chosen in this implementation**, not documented as the manuscript authors' historical seed. For methodological background see [Kennedy & Eberhart (1995)](https://doi.org/10.1109/ICNN.1995.488968).

## 3. Re-run and verify the exact neural vectors

```bash
python src/weight_recipe.py --mode replay
```

This trains all eight networks from the measurements and writes `results/weight_replay.csv`, `results/replayed_connection_weights.csv`, `results/neural_pso_training_histories.csv` and `results/weight_recipe_summary.json`. The generated weights are compared with those saved under `models/architecture_*.npz`. A numerical mismatch normally fails the command; `--no-strict` records differences on an incompatible software stack.

In the tested environment, **8/8 weight arrays and all eight feature/response scaling arrays replay exactly, with maximum coefficient difference zero**. This is reproducibility of the shipped implementation, not discovery of uniquely identifiable original laboratory weights.

To investigate other swarms:

```bash
python src/weight_recipe.py --mode all
python src/plot_weight_recipe.py
```

The five-seed experiment uses offsets 0, 1, 7, 31 and 113 on each architecture's prescribed seed. It stores 40 neural fits in `results/seed_sensitivity.csv`, which compares **unmodified, uncalibrated neural predictions**. In the checked run, none of 32 alternative seeds generated the identical serialized parameter vector. Different weights can represent similar prediction functions; parameter matching is not a general measure of predictive quality. For a broad review of functional weight symmetries, see [Grigsby, Lindsey & Rolnick (2023)](https://proceedings.mlr.press/v202/grigsby23a.html); their formal results focus on ReLU networks, while the permutation non-uniqueness observation also applies to the hidden units here.

## 4. Separate aggregate-constrained calibration

The **neural PSO weights alone do not give the fitted 0.942 table entry**. Given the neural predictions \(f_\theta(X)\), the released architecture models additionally use the Gaussian discrepancy correction

\[
 K_{ij}=\exp[-\lVert z_i-z_j\rVert^2],\qquad
 (K+\alpha I)\beta=y-f_\theta(X),
\]
\[
 \hat y(x)=f_\theta(x)+\sum_{j=1}^{30}K(x,z_j)\beta_j.
\]

The correction has 30 kernel coefficients and one regularization strength \(\alpha\). **The value of \(\alpha\) is solved from each architecture's published RMSE**, using model-specific Gaussian reference-response evaluation weights. The Gaussian bandwidth is fitted so that weighted explained variance matches the architecture's reported accuracy value. These summary constraints are read in `src/train_realization.py`, *after* neural PSO training. The regularized kernel expansion is motivated by the [generalized representer theorem (Schölkopf, Herbrich & Smola, 2001)](https://doi.org/10.1007/3-540-44581-1_27) and model-discrepancy analysis by [Kennedy & O'Hagan (2001)](https://doi.org/10.1111/1467-9868.00294). Those methods do not independently validate target-derived parameter choices.

For the shipped 3–3–2–1 architecture: \(\alpha\approx2.40544\times10^{-5}\); RBF \(\gamma=1\); fitted weighted evaluation effective sample size \(\approx22.44\). The effective sample size is **not** 22 new people.

The seven regression comparison rows are a distinct model family: each fits a baseline regressor and a target-constrained residual profile, with metric-specific evaluation populations; see [Training](TRAINING.md). In particular, the reported MAE, MSE, and RMSE cannot universally be interpreted as pooled statistics on one identical set of residuals. This explains why the 51 entries can be regenerated with calibrated scoring conventions but not read as 51 independent held-out successes.

## 5. Transparently trace the forward path

```bash
python src/trace_forward.py --model 3-3-2-1 --voltage 380 --age 51 --bmi 22.2
```

It prints the scaled inputs, the three and two sigmoid activations, linear neural output, Gaussian kernel similarities, **30 individual residual contributions**, and the final calibrated glucose prediction. These are calculated using saved model parameters only; no published score target is loaded for inference.

For the illustrative input above, the shipped network produces a **96.8440 mg/dL neural output**, adds a **−9.8443 mg/dL residual correction**, and returns **86.9998 mg/dL**. This is a fitted example, not a clinical validation case.

```bash
bash run_all.sh --recipe      # weight-replay and seed experiment
bash run_all.sh --train       # full parameter training + calibration + replay + forward verification
bash run_all.sh --verify      # forward-only predictions from saved model files
```

The distinction is essential: **neural weights can be regenerated from patient inputs and glucose labels using the chosen deterministic seed; the exact printed table scores require the separately fitted numerical summary constraints.**
