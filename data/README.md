# Published numerical data

The CSV files in this directory transcribe numerical values printed in the [TechRxiv manuscript](https://doi.org/10.36227/techrxiv.24465955.v1). They are inputs to this computational companion, not new experimental observations.

- `observed_30.csv`: 30 voltage, age, BMI, instrument and reference-glucose records from ten participants.
- `glucose_loading.csv`: the same participant group in wide form, with pre-loading, post-loading and random readings.
- `hba1c.csv`: ten additional published HbA1c-study records.
- `paper_architectures.csv`: eight published ANN architecture metric pairs used as constraints in the inverse-calibration experiment.
- `paper_algorithm_comparison.csv`: seven model-comparison rows, also used as inverse constraints.
- `paper_hardware.csv`: numerical component and optimizer parameters.
- `paper_prior_work_comparison.csv`: published baseline comparisons from other studies; no external patient data are included.

No fabricated measurement appears in `data/`. The separately modeled nine-row Clarke extension is stored in `results/clarke_conditional_population.csv` with its simulated/observed provenance labels.
