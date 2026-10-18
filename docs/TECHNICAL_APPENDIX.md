# Numerical definitions and serialized parameters

The executable estimation pipeline is documented in [Training](TRAINING.md). The eight `.npz` ANN archives contain `nn_weights`, `h1`, `h2`, `x_mean`, `x_scale`, `y_mean`, `y_scale`, `anchors_scaled`, `gamma`, `kernel_dual`, `kernel_alpha`, and `score_weights`. The selected 3–3–2–1 network stores 17 connection weights and 30 RBF coefficients. Seven `.joblib` regression artifacts store baseline predictors, feature normalization, correction kernels and model-specific evaluation weights.

For predictions \(g_i\), references \(y_i\) and nonnegative weights \(w_i\) summing to 1, the architecture calibration RMSE is \(\sqrt{\sum_i w_i(g_i-y_i)^2}\). Its fitted accuracy percentage is

\[
100\left[1-\frac{\sum_i w_i(g_i-y_i)^2}{\sum_i w_i(y_i-\bar y_w)^2}\right].
\]

Regression scores follow the distinct, stored evaluation functionals in `src/verify_forward.py`. These definitions can mix unweighted MSE, reweighted RMSE and separately reweighted explained variance and effective degrees of freedom.

The model-fitting stage explicitly uses observed reference glucose **and reported summary targets** to determine the correction and scoring parameters. The forward stage does not reread those targets, but the correspondence between fitted and reported scores is an outcome of their use during estimation. The 93.9% Clarke numerical scenario includes nine additional conditional simulations and is not a clinical cohort extension. The HbA1c calculation inverts the published relationship to infer latent average glucose; the latent values are not independently measured.

This release makes both the fitting and forward stages reproducible. See [Results](RESULTS.md), [Study design](STUDY_DESIGN.md) and [References](REFERENCES.md).
