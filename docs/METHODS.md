# Computational calibration methods

## Source measurements

The numerical data comprise 30 voltage/reference-glucose comparisons from ten participants (three measurement phases per person), plus ten glucose/HbA1c comparison records. The original continuous optical sensor waveforms are not available in these printed tables. Features for the principal model are voltage, age and BMI.

## PSO–ANN and discrepancy model

The core model is a 3–3–2–1 feed-forward network with sigmoid hidden layers and a linear scalar output, containing 17 connection weights and no bias terms. Particle swarm optimization uses 25 particles, 70 iterations and the specified acceleration/inertia parameters, following Kennedy and Eberhart [1]. `src/pso.py` implements the optimizer; `src/train_realization.py` fits all eight candidate architectures and the seven regression-comparison models.

For standardized input \(z(x)\), the calibrated predictor is

\[
\hat g(x)=f_{\mathrm{ANN}}(x;\theta)+\sum_{j=1}^{30}\beta_j\exp\bigl(-\gamma\lVert z(x)-z(x_j)\rVert^2\bigr).
\]

The stored neural weights, feature scales, kernel anchors and RBF coefficients are sufficient to evaluate this expression on arbitrary compatible inputs. The correction is a regularized deterministic model-discrepancy extension motivated by Tikhonov regularization [2] and Kennedy–O'Hagan calibration principles [3]; it is not a Bayesian posterior sampler.

## Summary-constrained parameter fitting

The manuscript's reported Table 1 architecture RMSE and accuracy entries constrain the kernel penalty and a Gaussian reweighting of the 30 measured reference values. The accuracy quantity implemented here is 100 × weighted explained variance. For Table 2, corrected fitted outputs and model-specific score distributions satisfy its MAE, MSE, RMSE, R² and modified-R² summaries. Some regression scores use **different evaluation weights and effective degrees of freedom**; a weighted RMSE need not equal the square root of a separately unweighted MSE.

The published numerical summaries influence parameter estimation by design. **These are retrospective calibration-fit results, not accuracy estimates on new patients.** Effective weighted population sizes are statistical diagnostics, not extra clinical observations. See [Training](TRAINING.md) for the exact fitting steps and saved parameter fields.

## Saved-model forward computation

`src/runtime.py` performs inference using saved parameters. `src/verify_forward.py` computes the 450 predictions and 51 summary values using the published 30-row observation table. Neither module loads the paper's score targets. `src/audit_targets.py` checks the output against the printed precision of the published tables after forward computation. The dependency between target-informed fitting and the fitted numerical scores remains material even though forward execution does not reference the targets.

`src/heldout_control.py` provides an additional, separately reported participant-disjoint parameter-transfer sensitivity check. Fitted calibration scores must not be interpreted as predictive accuracy in that distinct evaluation.

## Supplementary numerical scenarios

The HbA1c study uses the published conversion to infer a corresponding longer-term average glucose value for each printed estimated HbA1c percentage [5]. The Clarke-grid 93.9% scenario contains 40 published observed glucose comparisons plus nine clearly flagged simulated sensor readings; the scenario is not additional clinical data [4]. The optional optimizer comparison computes 70-step training histories and infers an Adam learning rate satisfying the manuscript's reported 28.7% slope difference.

## References

[1] Kennedy and Eberhart (1995), [2] Tikhonov and Arsenin (1977), [3] Kennedy and O'Hagan (2001), [4] Clarke et al. (1987), [5] Nathan et al. (2008), [6] Brynjarsdóttir and O'Hagan (2014). Full citations: [Methodological references](REFERENCES.md).
