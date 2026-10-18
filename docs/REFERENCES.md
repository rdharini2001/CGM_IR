# Methodological references

The OPTIC implementation follows the optical sensing and network specifications of the accompanying manuscript. The supplementary calibration and numerical analyses use established ideas from optimization, regularization, inverse problems, distribution weighting, and clinical glucose assessment. These references provide the mathematical background for the methods; they do not independently establish the clinical performance of the fitted models.

## Optimization and calibration

1. **Kennedy, J., and Eberhart, R. (1995).** [Particle swarm optimization](https://doi.org/10.1109/ICNN.1995.488968). *Proceedings of ICNN '95*, 4, 1942–1948. The foundation of the particle-based optimizer used to fit the serialized ANN weights. The optimizer implementation and fitted weight archives are both included.
2. **Tikhonov, A. N., and Arsenin, V. Y. (1977).** *Solutions of Ill-Posed Problems*. V. H. Winston & Sons. [Library record](https://search.worldcat.org/title/2837312). The regularization principle behind the kernel residual correction, including its penalty parameter. This program implements a deterministic regularized correction; it is **not** a fully Bayesian posterior inference procedure.
3. **Kennedy, M. C., and O'Hagan, A. (2001).** [Bayesian calibration of computer models](https://doi.org/10.1111/1467-9868.00294). *Journal of the Royal Statistical Society: Series B*, 63(3), 425–464. The distinction between a primary computational model and an explicit model-discrepancy term motivates the ANN-plus-residual architecture. Our implementation uses an RBF correction rather than their complete Bayesian model.
4. **Hansen, L. P. (1982).** [Large sample properties of generalized method of moments estimators](https://doi.org/10.2307/1912775). *Econometrica*, 50(4), 1029–1054. Moment-constraint estimation supplies an analogy for fitting a numerical model to aggregate statistics. The use of *published evaluation endpoints as fitting constraints* in this repository is a bespoke inverse numerical construction; the cited theory does not turn those fitted scores into an unbiased accuracy estimate.

## Scoring and predictive evaluation

5. **Sugiyama, M., Krauledat, M., and Müller, K.-R. (2007).** [Covariate shift adaptation by importance weighted cross validation](https://www.jmlr.org/papers/v8/sugiyama07a.html). *Journal of Machine Learning Research*, 8, 985–1005. Background on distribution-weighted scoring. In this repository the scoring populations are *inferred using the published metrics*, not estimated from a verified external target population; therefore the importance-weighting guarantees of Sugiyama and colleagues **do not apply** to these calibration-fit scores.
6. **Varma, S., and Simon, R. (2006).** [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91). *BMC Bioinformatics*, 7, 91. Clarifies why scores on observations or evaluation distributions used to select model settings are not unbiased held-out validation scores. This edition reports calibration-fit metrics rather than claims of held-out clinical accuracy.
7. **Friedman, J., Hastie, T., and Tibshirani, R. (2010).** [Regularization paths for generalized linear models via coordinate descent](https://doi.org/10.18637/jss.v033.i01). *Journal of Statistical Software*, 33(1), 1–22. Methodological reference for ElasticNet in the model-comparison experiments.

## Glucose measurements and clinical context

8. **Clarke, W. L., Cox, D., Gonder-Frederick, L. A., Carter, W., and Pohl, S. L. (1987).** [Evaluating clinical accuracy of systems for self-monitoring of blood glucose](https://doi.org/10.2337/diacare.10.5.622). *Diabetes Care*, 10(5), 622–628. Defines the clinically interpreted glucose error grid. The measured pairs and the separately marked simulated grid extension must not be treated as the same evidentiary population.
9. **Nathan, D. M., et al. (2008).** [Translating the A1C assay into estimated average glucose values](https://doi.org/10.2337/dc08-0545). *Diabetes Care*, 31(8), 1473–1478. Provides the average-glucose and HbA1c conversion relationship. An inferred long-term mean is an algebraic input, not a measured longitudinal glucose trace.

## Study design, numerical scenarios and transparent reporting

10. **Brynjarsdóttir, J., and O'Hagan, A. (2014).** [Learning about physical parameters: the importance of model discrepancy](https://doi.org/10.1088/0266-5611/30/11/114007). *Inverse Problems*, 30(11), 114007. A direct treatment of the inverse-calibration identifiability problem: parameter values and discrepancy effects can be confounded. This supports treating recovered calibration coefficients as fitted computational parameters rather than uniquely identified tissue or sensor properties.
11. **Hainmueller, J. (2012).** [Entropy balancing for causal effects: a multivariate reweighting method to produce balanced samples in observational studies](https://doi.org/10.1093/pan/mpr025). *Political Analysis*, 20(1), 25–46. Demonstrates how reweighting can impose exact moment constraints. The present score-conditioned weights are a mathematical analogy only; we do not claim causal identification, covariate balance, or population representativeness.
12. **Morris, T. P., White, I. R., and Crowther, M. J. (2019).** [Using simulation studies to evaluate statistical methods](https://doi.org/10.1002/sim.8086). *Statistics in Medicine*, 38(11), 2074–2102. A framework for reporting numerical scenarios and simulation assumptions. This motivates explicit simulated-row labels and separation of the 93.9% scenario from the observed glucose comparisons.
13. **Collins, G. S., et al. (2024).** [TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods](https://doi.org/10.1136/bmj-2023-078378). *BMJ*, 385, e078378. Supports explicit descriptions of development, evaluation populations, and intended use. We cite it as reporting guidance, not as an endorsement of the reconstructed performance estimates or a claim that the computational study has undergone clinical validation.

## Citing this project

Please cite the accompanying study: **Raghavan et al. (2023)**, [*Non-Invasive Continuous Glucose Monitoring using Near Infrared Sensors and PSO-ANN Algorithm*](https://doi.org/10.36227/techrxiv.24465955.v1). Complete citations are available in [`REFERENCES.bib`](../REFERENCES.bib).

### Weight training, kernel expansions and identifiability

- **Schölkopf, Herbrich & Smola (2001)**, *A Generalized Representer Theorem*, COLT, pp. 416–426. [DOI](https://doi.org/10.1007/3-540-44581-1_27). Supports a finite training-anchor expansion for regularized kernel methods; it does not determine a particular manuscript-dependent regularization parameter.
- **Grigsby, Lindsey & Rolnick (2023)**, *Hidden Symmetries of ReLU Networks*, ICML, PMLR 202. [Paper](https://proceedings.mlr.press/v202/grigsby23a.html). Formalizes redundancies in neural-network weight parameterizations (specific proofs concern ReLU models); cited for the general caution that replaying a particular weight array differs from uniquely identifying a predictive function or authors' unpublished weights.
