# Psychometric Audit Pipeline

## Overview

**Psychometric Audit Pipeline** is a research-grade, open-source framework for auditing the **measurement validity, reliability, fairness, and uncertainty** of mental-health self-report scale data prior to modeling. The project is motivated by a core methodological concern in applied machine learning and psychological research: models are frequently fit to questionnaire data without first establishing whether the underlying measurement properties justify such modeling.

This pipeline operationalizes an **audit-before-modeling** paradigm, integrating psychometric diagnostics, fairness screening, uncertainty quantification, and interpretable latent trait modeling within a single, reproducible workflow. The emphasis is on methodological rigor, transparency, and ethical restraint rather than predictive accuracy or deployment.

---

## Conceptual Framework

The pipeline is structured around four sequential questions:

1. **Is the data suitable for modeling?**  
   (Data integrity, response behavior, missingness)

2. **Is the scale internally consistent?**  
   (Reliability and item functioning)

3. **Is the scale comparable across groups?**  
   (Fairness and bias diagnostics)

4. **If so, what latent structure can be estimated transparently?**  
   (Interpretable AI via IRT)

Each stage produces explicit outputs and warnings, and later stages are designed to be interpretable only in light of earlier audit results.

---

## Data Quality Audit

The pipeline begins with item-level diagnostics, including:

- **Missingness analysis** to quantify nonresponse patterns  
- **Response distribution summaries** for exploratory inspection  
- **Floor and ceiling effect detection** to identify restricted variability  
- Explicit handling of sentinel missing codes (e.g., `-1 → NA`)

These checks are intended to surface structural issues that would invalidate reliability or latent modeling if ignored.

---

## Reliability and Internal Consistency

Internal consistency is evaluated using:

- **Cronbach’s alpha** for the full scale  
- **Corrected item–total correlations** to identify problematic items  
- Group-stratified reliability estimates where applicable  

To avoid overinterpretation of point estimates, the pipeline optionally supports **bootstrap-based confidence intervals** for reliability metrics. Minimum sample-size safeguards are enforced to prevent invalid subgroup estimates.

---

## Fairness and Bias Diagnostics

Fairness is evaluated through **Differential Item Functioning (DIF) screening**, implemented using regression-based methods that test whether item responses differ across groups after conditioning on total score.

Key features include:
- Support for multiple grouping variables (e.g., gender, age group, site/region)
- Explicit minimum sample-size thresholds per group
- Separation of overall scale differences from item-specific bias

An optional **bootstrap DIF stability procedure** estimates how consistently items are flagged under resampling, emphasizing robustness and reproducibility rather than binary decisions.

---

## Interpretable AI Modeling (Item Response Theory)

When audit conditions are satisfied, the pipeline supports **interpretable AI modeling** via a Bayesian **Item Response Theory Graded Response Model (GRM)** implemented in PyMC.

The model estimates:
- Item discrimination parameters  
- Ordered threshold parameters  
- Latent severity scores (θ) for respondents in the fitted subset  

The approach is explicitly **black-box free**:
- All learned parameters are saved as transparent tables
- Reusable scoring utilities enable evaluation of new data without retraining
- Item and test **information functions** quantify precision across the latent continuum

Model fitting is performed on configurable subsets of complete data to balance statistical fidelity and computational feasibility.

---

## Computational Design and Reproducibility

- Fully configuration-driven (`config.yaml`)
- Deterministic sampling with fixed random seeds
- Modular codebase enabling partial or staged execution
- Designed to scale to large datasets through controlled sampling
- Explicit separation between audit procedures and modeling procedures

---

## Scope, Ethics, and Limitations

This project is intended for **research and educational use only**. It is not designed for clinical diagnosis, individual-level decision making, or operational deployment. The pipeline includes refusal logic and warnings when assumptions are violated (e.g., insufficient group sizes).

---

## Forking and Extension

Users are encouraged to fork the repository to:
- Adapt the pipeline to new instruments or datasets
- Tune computational trade-offs for available hardware
- Extend planned modules (e.g., IRT-DIF, measurement invariance)
- Integrate additional reporting or audit components

Contributors are encouraged to preserve the project’s ethical framing, safeguards, and emphasis on uncertainty-aware interpretation.

---

## License

This project is released under the MIT License.
