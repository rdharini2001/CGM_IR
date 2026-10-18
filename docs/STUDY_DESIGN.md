# Study design

## Scope

OPTIC is a **retrospective computational calibration case study** using published observations and manuscript-level numerical constraints. The release implements both the complete parameter-fitting workflow and a separately callable saved-model forward predictor. It investigates which PSO–ANN, discrepancy-correction and scoring parameters realize the printed result tables and makes those fitted parameterizations reproducible.

The study does not claim to have acquired additional optical signals or enrolled new participants, and its fitted scores are not independent clinical-validation estimates.

## Source populations

Ten participants each contribute three observations (before loading, after loading and random), giving 30 measured voltage/reference-glucose pairs. Another ten participants contribute glucose/HbA1c comparisons but do not have the required complete three-input ANN records in the printed table. The **same 30 labeled observations** are used for fitting the model and evaluating the numerical calibration endpoints.

The architecture-specific weighted scoring distributions, with possible non-integer effective sample size, are mathematical weights on existing observations, **not additional patients**. The Clarke-grid extension combines 40 published observed glucose comparisons with nine **separately labeled simulations**, rather than an expanded measured patient cohort.

## Training, inference and interpretation

Training (`src/train_realization.py`) fits the paper's candidate ANN architectures with PSO, plus RBF discrepancy correction and model-specific scoring parameters constrained by the published scores. The regression comparison applies baseline fitting and analogous calibrated scoring. The artifacts in `models/` contain the recovered ANN weights, kernel coefficients and necessary scalers.

Forward inference (`src/runtime.py`) requires only the saved artifacts and model inputs, and numerical verification (`src/verify_forward.py`) does not access the manuscript targets. This modular structure enables independent inspection of fitted coefficients and scoring arithmetic; it does **not** make the target-conditioned fitted outputs into held-out accuracy estimates.

Participant-disjoint transfer is separately assessed by `src/heldout_control.py`, which inherits a penalty estimated on the full calibration data. Its results are a sensitivity diagnostic rather than independent model selection.

## Literature basis

The work draws on [Kennedy and Eberhart (1995)](https://doi.org/10.1109/ICNN.1995.488968) for particle-swarm optimization, [Kennedy and O'Hagan (2001)](https://doi.org/10.1111/1467-9868.00294) for computational model discrepancy, [Brynjarsdóttir and O'Hagan (2014)](https://doi.org/10.1088/0266-5611/30/11/114007) for parameter-identifiability considerations, [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086) for clear computational simulation reporting, and [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) for distinguishing model development from validation. These principles do not establish accuracy on an independent clinical cohort.

See [Training](TRAINING.md), [Methods](METHODS.md) and [Results](RESULTS.md).

## Deterministic neural training versus inverse calibration

The neural weight-training protocol uses only the manuscript's 30 printed input/reference pairs. Its fixed implementation seed allows exact replay of the distributed *OPTIC* neural parameters; it does **not** establish those as the manuscript authors' undocumented original weights. Architecture-targeted RBF regularization and scoring-population fitting are downstream and separately use the printed model-summary metrics. The [weight protocol and seed experiment](WEIGHT_RECIPE.md) document these two distinct information flows.
