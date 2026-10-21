# References and methodological rationale

The numbered references below identify the source supporting each modeling or evaluation choice. The first citation is the accompanying research article. The supplemental methods use the same identifiers.

[1] Suma K. V., Dharini Raghavan, Maya V. Karki, Narayana Sharma, and Gundu H. R. Rao. *Non-Invasive Continuous Glucose Monitoring using Near Infrared Sensors and PSO-ANN Algorithm.* TechRxiv (2023). https://doi.org/10.36227/techrxiv.24465955.v1. **Original optical architecture, inputs, PSO specification, Table 1/Table 2, clinical observations.**

[2] ASTM International. *ASTM E1655-17(2024): Standard Practices for Infrared Multivariate Quantitative Analysis.* https://store.astm.org/e1655-17r24.html. **Supports infrared calibration model development as a defined analytical stage; distinguishes continuing validation on unknown samples.** This standard does not claim that training-set RMSE equals external validation performance.

[3] Cozzolino D. et al. *Near-infrared spectroscopy and hyperspectral imaging: non-destructive analysis of biological materials.* Chemical Society Reviews (2014), DOI:10.1039/C4CS00062E. https://doi.org/10.1039/C4CS00062E. **Calibration error, cross-validation, independent prediction and reporting of analytical accuracy as distinct estimands.**

[4] Sämann A. et al. *Non-invasive blood glucose monitoring by means of near infrared spectroscopy: investigation of long-term accuracy and stability.* Experimental and Clinical Endocrinology & Diabetes (2000). DOI:10.1055/s-2000-8137. https://pubmed.ncbi.nlm.nih.gov/11026754/. **Individualized spectral calibration in a near-infrared glucose study; transfer/stability is a separate validation question.**

[5] Kennedy J. and Eberhart R. *Particle swarm optimization.* Proceedings of ICNN (1995), DOI:10.1109/ICNN.1995.488968. https://doi.org/10.1109/ICNN.1995.488968. **Particle position, velocity, personal best and global best.**

[6] Schölkopf B., Herbrich R. and Smola A. J. *A Generalized Representer Theorem.* COLT (2001), DOI:10.1007/3-540-44581-1_27. https://mlanthology.org/colt/2001/scholkopf2001colt-generalized/. **Regularized kernel expansion for residual calibration.**

[7] Golub G. H., Heath M. and Wahba G. *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter.* Technometrics 21:215-223 (1979), DOI:10.1080/00401706.1979.10489751. https://doi.org/10.1080/00401706.1979.10489751. **Data-driven regularization sensitivity analysis, not parameter choice from printed article values.**

[8] Clarke W. L. et al. *Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose.* Diabetes Care 10:622-628 (1987), DOI:10.2337/diacare.10.5.622. https://doi.org/10.2337/diacare.10.5.622. **Clinical error-grid framework for paired measurements.**

[9] Nathan D. M. et al. *Translating the A1C Assay Into Estimated Average Glucose Values.* Diabetes Care 31:1473-1478 (2008), DOI:10.2337/dc08-0545. https://pmc.ncbi.nlm.nih.gov/articles/PMC2742903/. **Long-term estimated mean glucose relation to HbA1c: eAG = 28.7 HbA1c - 46.7 mg/dL.**

[10] Rinnan Å., van den Berg F. and Engelsen S. B. *Review of the most common pre-processing techniques for near-infrared spectra.* TrAC Trends in Analytical Chemistry 28:1201-1222 (2009), DOI:10.1016/j.trac.2009.07.007. https://doi.org/10.1016/j.trac.2009.07.007. **Optical preprocessing background.** This is a broad full-spectrum NIR reference; our three-feature single-channel voltage model does not apply those full-spectrum techniques directly.

[11] Review of Non-Invasive Glucose Sensing Techniques: Optical, Electrical and Breath Acetone. Sensors (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7085605/. **Effects of tissue variability and interference on optical glucose sensing; motivation for distinguishing discovery and inter-individual transfer.**

## Interpretation used throughout the companion

- **Discovery calibration:** fitted (in-sample) statistical agreement on the transcribed 30 paired observations, appropriate to report as a calibration result under ASTM E1655 [2] and the NIR chemometrics literature [3].
- **Participant-held-out and phase-held-out:** complementary transfer analyses, not historical claims about which protocol the original authors used [3,4,11].
- **Paper performance thresholds:** numerical values printed in [1], used only after model fitting to form the descriptive `analysis/calibration_coverage.py` audit. For the eight architecture **accuracy** percentages, our provisional comparison uses 100 R²; it is a proxy, not a guaranteed reproduction of the manuscript's accuracy definition.
- **Kernel-corrected calibration:** an additional computational extension justified by [6,7], described in the supplementary methods rather than claimed to be a formula present in [1].
- **Clinical observations:** 30 glucose-loading pairs and a distinct ten-participant HbA1c series transcribed from [1]. The article's 93.9% Clarke Zone A may represent a population not listed row by row; the 30-pair subset gives 100% Zone A under the standard boundary. Neither population is silently substituted for the other.
