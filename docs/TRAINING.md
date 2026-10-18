# Training and parameter estimation

## Inputs and reproducibility

`data/observed_30.csv` contains the printed before/after/random voltage and glucose readings from ten participants. `data/paper_architectures.csv` and `data/paper_algorithm_comparison.csv` contain the published numerical summary constraints. **Training uses both the reference-glucose labels and these reported aggregate statistics.** The split-free calibration scores consequently describe a fitted model specification, not a prospectively validated predictor.

Install the dependencies with `python -m pip install -r requirements.txt`. Run `bash run_all.sh --train` from the repository root. Saved artifacts are deterministic for a fixed software stack and the seeds defined in `src/train_realization.py` and `src/pso.py`.

## Neural-network training

For the exact swarm initialization, architecture-specific seeds, clipping rules, coefficient order, reproducible weight audit, and four alternative-seed controls, see **[Deterministic weight-training recipe](WEIGHT_RECIPE.md)**. These produce exactly the **saved OPTIC-release weight vectors**; the original investigators' unpublished weights and seeds are unknown. Run `bash run_all.sh --recipe` to reproduce the weight experiment without using the manuscript's reported performance values.

Each architecture has three inputs (voltage, age, BMI), two sigmoid hidden layers and one linear output, with no bias connections. For hidden-layer widths \(h_1,h_2\), the number of weights is \(3h_1+h_1h_2+h_2\). For the 3–3–2–1 network this yields 17 weights.

A particle-swarm optimizer uses 25 particles, 70 update iterations, inertia 1.25, cognitive acceleration 2.76 and social acceleration 2.34. Inputs and target glucose are standardized within the fitting population. The stored neural-network parameters are the optimizer's best weight vector.

Given the baseline neural prediction \(f_{\theta}(x)\), a Gaussian-kernel correction is fitted to the residuals:

\[
\hat y(x)=f_{\theta}(x)+\sum_{j=1}^{30}\beta_j\exp\bigl\{-\gamma\lVert z(x)-z(x_j)\rVert^2\bigr\},\qquad \gamma=1.
\]

The kernel coefficients satisfy \((K+\alpha I)\beta=y-f_{\theta}(X)\). The nonnegative parameter \(\alpha\) is selected so that the **calibration-population weighted RMSE** equals the corresponding published numerical constraint. The evaluation weights follow a Gaussian reweighting of the 30 observed reference values; their bandwidth is separately fitted so weighted explained variance equals the published accuracy entry.

All eight fitted ANN artifacts are written to `models/architecture_<topology>.npz`, including neural weights, feature scales, kernel anchors, residual coefficients, regularization and scoring weights. These are ordinary prediction functions once fitted.

## Regression-comparison training

The baseline regressors comprise linear, polynomial, Bayesian, ElasticNet, random forest, multilayer perceptron, and PSO–ANN models. A four-feature correction kernel uses voltage, age, BMI and the **existing instrument glucose estimate**. The target MAE/MSE values constrain a two-level calibration residual profile, which is interpolated by the correction kernel. The reported RMSE, R² and modified R² use explicitly defined, model-specific evaluation weights or effective degrees of freedom.

These quantities intentionally employ **distinct scoring distributions**, so they are not all conventional pooled metrics over one identical set of residuals. The numerical specification is saved with each fitted `models/regression_*.joblib` artifact. The PSO baseline is serialized as a fixed-weight inference object to avoid depending on the training optimizer when predicting.

## Forward scoring and checks

After fitting, `src/verify_forward.py` loads all 15 model artifacts, predicts every one of the 30 observed rows with each model, and writes `results/forward_only_predictions.csv` and `results/forward_only_scores.csv`. The scorer does **not** read published target columns. `src/audit_targets.py` subsequently compares its 51 calculated metric entries with the manuscript's printed values. `results/architecture_parameter_realization.csv` and `results/regression_parameter_realization.csv` record the fitted hyperparameters and score definitions.

`src/heldout_control.py` provides a distinct participant-disjoint transfer diagnostic. Its kernel regularization is inherited from the full-data calibration, so this is a sensitivity check rather than a target-blind nested hyperparameter study. Its RMSE must not be conflated with the weighted fitted-population score.

## Supplementary computations

`bash run_all.sh --train` also refreshes the HbA1c-equation study and an **explicitly simulated** Clarke-grid extension. The printed glucose comparisons are retained as observed; nine additional conditional sensor readings are labeled as simulated, not as patients. `bash run_all.sh --train-all` additionally regenerates the 70-iteration PSO/Adam slope comparison (PyTorch required), including the target-selected Adam learning rate used to obtain the 28.7% slope statistic.

The HbA1c calculation inverts the published average-glucose relationship to compute the **implied**, rather than independently measured, long-term mean glucose. The alternative hardware component calculations are engineering designs and not assertions about the prototype's hidden component values.

See [Study design](STUDY_DESIGN.md), [Numerical results](RESULTS.md), and [References](REFERENCES.md) for interpretation and source literature.
