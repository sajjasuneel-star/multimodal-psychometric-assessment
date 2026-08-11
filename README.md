https://doi.org/10.5281/zenodo.21887510

# Multimodal NLP Framework for Automated Psychometric Assessment in Collaborative Learning

This repository provides a complete **Python/PyTorch implementation** of a multimodal deep learning framework for automated psychometric assessment in collaborative learning environments.

The framework is designed to process heterogeneous collaborative interaction evidence including:

* Textual discourse
* Acoustic descriptors
* Behavioural interaction logs
* Optional visual-affect cues

It integrates:

* Modality-specific encoders
* Cross-modal attention
* Availability-aware modality gating
* Temporal aggregation
* Multitask psychometric prediction
* Uncertainty estimation
* Calibration
* Explainability
* Adaptive intervention support

The implementation follows a **simulation-based experimental protocol** and supports reproducible evaluation of continuous psychometric traits, ordinal trait levels, collaborative outcomes, temporal consistency, uncertainty quality, psychometric measurement properties, multimodal robustness, and statistical significance.

---

## 1. Research Objective

The primary objective of this implementation is to develop and evaluate a **measurement-aware multimodal learning framework** capable of estimating temporally evolving psychometric characteristics from collaborative-learning interaction data.

The framework estimates the following five psychometric constructs:

1. Engagement
2. Collaboration Quality
3. Socio-Emotional Presence
4. Cognitive Participation
5. Participation Balance

Each construct is represented as a continuous score in the range **[0,1]** and is additionally mapped to an ordinal level:

* Low
* Moderate
* High

The system also predicts an auxiliary collaborative outcome and provides predictive uncertainty estimates.

---

## 2. Framework Overview

The complete processing pipeline is:

```text
Synthetic Collaborative-Learning Sessions
                    ↓
Multimodal Observation Generation
                    ↓
Text + Audio + Behavioural Logs + Visual Cues
                    ↓
Preprocessing and Temporal Alignment
                    ↓
Modality-Specific Encoding
                    ↓
Cross-Modal Multi-Head Attention
                    ↓
Availability-Aware Learned Modality Gating
                    ↓
Window-Level Multimodal Fusion
                    ↓
BiGRU-Based Temporal Aggregation
                    ↓
Multitask Psychometric Prediction
    ├── Continuous Trait Regression
    ├── Ordinal Trait Prediction
    ├── Auxiliary Outcome Prediction
    └── Heteroscedastic Uncertainty
                    ↓
Temporal Stability and Calibration
                    ↓
Integrated Gradients Explainability
                    ↓
Modality Contribution Analysis
                    ↓
Uncertainty-Aware Adaptive Intervention
```

---

## 3. Key Features

* Reproducible synthetic collaborative-learning session generation
* Five-dimensional latent psychometric state modelling
* Temporally correlated psychometric trait evolution
* Independent feature and reference-label generation
* Synthetic text discourse generation
* Acoustic feature simulation
* Behavioural interaction-log simulation
* Optional visual-affect feature simulation
* Controlled observation noise
* Missing-modality simulation
* Transformer-based text representation
* Modality-specific neural encoders
* Multi-head cross-modal attention
* Availability-aware modality masking
* Learned modality gating
* BiGRU temporal aggregation
* Attention-based temporal pooling
* Continuous psychometric regression
* Ordinal psychometric classification
* Auxiliary collaborative-outcome prediction
* Heteroscedastic predictive uncertainty
* Temporal-consistency regularization
* Contrastive representation pretraining
* Calibration analysis
* Integrated Gradients explainability
* Modality contribution analysis
* Rule-based adaptive intervention generation
* Cohort-wise cross-validation
* Three-seed repeated evaluation
* Baseline comparison
* Ablation experiments
* Stability-coefficient sensitivity analysis
* Missing-modality robustness analysis
* Synthetic-label leakage diagnostics
* Psychometric reliability and validity analysis
* Statistical significance testing
* Automated result export and visualization

---

## 4. Synthetic Dataset Configuration

The implementation generates a controlled multimodal collaborative-learning environment with the following default experimental settings:

| Parameter                | Value         |
| ------------------------ | ------------- |
| Total sessions           | 120           |
| Cohorts                  | 3             |
| Sessions per cohort      | 40            |
| Learners per session     | 3–5           |
| Session duration         | 15–30 minutes |
| Temporal window          | 5 seconds     |
| Psychometric traits      | 5             |
| Trait range              | [0,1]         |
| Initial trait mean       | 0.50          |
| Initial trait SD         | 0.15          |
| Temporal persistence     | 0.85          |
| Process noise SD         | 0.03          |
| Observable noise         | 5–15%         |
| Missing text             | 0%            |
| Missing audio            | 5%            |
| Missing behavioural logs | 8%            |
| Missing visual cues      | 20%           |
| Random seeds             | 42, 123, 2026 |

The synthetic design preserves causal separation between latent psychometric states, observable features, and evaluation labels.

Conceptually:

```text
Latent Psychometric State T
        ├────────→ Observable Features X
        │
        └────────→ Reference Labels Y
```

No direct `Y → X` pathway is used. This reduces the risk of artificially inflated prediction performance caused by deterministic label embedding.

---

## 5. Psychometric Constructs

### 5.1 Engagement

Represents sustained behavioural and cognitive involvement in the collaborative activity.

Example indicators include:

* Contribution frequency
* Response activity
* Task-oriented discourse
* Speaking activity
* Interaction continuity
* Attention-related signals

### 5.2 Collaboration Quality

Represents reciprocal communication, coordination, knowledge sharing, and constructive joint problem solving.

Example indicators include:

* Turn-taking
* Clarification
* Agreement
* Coordination events
* Response latency
* Consensus-building patterns

### 5.3 Socio-Emotional Presence

Represents affective expression, emotional regulation, and supportive interpersonal interaction.

Example indicators include:

* Affective discourse
* Prosodic variation
* Supportive responses
* Disagreement patterns
* Conflict cues
* Emotion-regulation indicators

### 5.4 Cognitive Participation

Represents cognitively meaningful contribution to collaborative knowledge construction.

Example indicators include:

* Explanations
* Questions
* Reasoning
* Justification
* Clarification
* Reflection
* Task-relevant concepts

### 5.5 Participation Balance

Represents equitable distribution of contribution opportunities across group members.

Example indicators include:

* Contribution ratios
* Speaking-time distribution
* Turn counts
* Dominance indicators
* Response opportunities
* Participation regularity

---

## 6. Ordinal Trait Mapping

Continuous psychometric scores are mapped to three ordered levels:

| Level    | Score Range         |
| -------- | ------------------- |
| Low      | 0.00 ≤ score < 0.33 |
| Moderate | 0.33 ≤ score < 0.67 |
| High     | 0.67 ≤ score ≤ 1.00 |

The thresholds are fixed before model training.

---

## 7. Repository Structure

```text
multimodal_psychometric/
│
├── run_all.py
├── requirements.txt
├── README.md
│
├── mmpa/
│   ├── config.py
│   ├── data_generation.py
│   ├── dataset.py
│   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   ├── temporal.py
│   │   ├── heads.py
│   │   ├── proposed_model.py
│   │   └── baselines.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── pretraining.py
│   │   ├── trainer.py
│   │   └── cross_validation.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── uncertainty.py
│   │   ├── leakage.py
│   │   ├── psychometrics.py
│   │   ├── statistics.py
│   │   ├── ablation.py
│   │   └── sensitivity.py
│   │
│   ├── explainability/
│   │   ├── integrated_gradients.py
│   │   ├── modality_importance.py
│   │   └── intervention.py
│   │
│   └── utils/
│       ├── reproducibility.py
│       ├── io.py
│       └── plotting.py
│
├── notebooks/
│   └── multimodal_psychometric_colab.ipynb
│
└── outputs/
    ├── datasets/
    ├── models/
    ├── predictions/
    ├── metrics/
    ├── tables/
    └── figures/
```

The exact folder structure may vary slightly depending on the packaged version, but the implementation follows the same modular organization.

---

## 8. Installation

### Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd multimodal_psychometric
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Running the Implementation

### Quick Validation Mode

Use this mode to verify that the entire pipeline works correctly with reduced computational requirements.

```bash
python run_all.py --mode quick --output-dir outputs_quick
```

Quick mode is recommended for:

* Installation verification
* Debugging
* CPU execution
* Code inspection
* CI testing
* Preliminary experimentation

### Manuscript-Scale Experimental Mode

Use:

```bash
python run_all.py --mode paper --output-dir outputs_paper
```

This mode uses the manuscript-scale experimental configuration, including:

* 120 collaborative sessions
* 3 cohorts
* Full multimodal architecture
* Repeated random seeds
* Cohort-wise validation
* Baseline comparisons
* Ablation analysis
* Uncertainty analysis
* Statistical evaluation

> **Note:** GPU execution is strongly recommended.

---

## 10. Google Colab

A Colab-oriented notebook is included in the repository.

The recommended runtime is:

```text
Runtime → Change runtime type → GPU
```

Then install dependencies and execute the notebook cells sequentially.

The notebook follows the complete pipeline from synthetic data generation to experimental evaluation.

---

## 11. Modality-Specific Encoders

### Text Encoder

The text branch uses a Transformer-based semantic representation followed by projection into the common multimodal embedding space.

```text
Collaborative Utterance
        ↓
Transformer Encoder
        ↓
Contextual Text Representation
        ↓
Projection Layer
        ↓
Text Embedding
```

### Acoustic Encoder

The acoustic stream uses simulated paralinguistic descriptors such as:

* Speaking rate
* Pause duration
* Speaking-turn duration
* Pitch variation
* Intensity variation
* Overlap frequency

These descriptors are encoded using a neural projection network.

### Behavioural Encoder

Behavioural logs contain temporally organized interaction events such as:

* Message frequency
* Document edits
* Turn-taking
* Response latency
* Contribution ratios
* Alternation patterns
* Coordination events

A temporal neural encoder produces the behavioural representation.

### Visual Encoder

The optional visual modality uses high-level affect and attention descriptors rather than raw images.

Possible features include:

* Affect intensity
* Attention
* Gaze stability
* Positive affect
* Negative affect
* Facial activation

Visual information is optional and can be unavailable without preventing model inference.

---

## 12. Cross-Modal Fusion

For every synchronized interaction window, the available modality embeddings are represented as multimodal tokens.

```text
Text
Audio
Logs
Visual
  ↓
Multi-Head Cross-Modal Attention
```

The model then applies availability-aware learned gating.

Cross-modal attention and gating serve different functions.

### Attention

Models information exchange among modalities.

### Gating

Determines the relative contribution of each available modality.

Unavailable modalities are masked before normalization.

The fused representation for a temporal window is conceptually:

```text
Z(w) = Σ α(w,m) H(w,m)
```

where `α(w,m)` represents the learned contribution of modality `m` at window `w`.

---

## 13. Temporal Aggregation

Window-level multimodal representations are processed sequentially.

The default implementation uses:

```text
Fused Window Embeddings
        ↓
Bidirectional GRU
        ↓
Temporal Attention Pooling
        ↓
Session Representation
```

This enables the framework to model the evolution of collaborative behaviour over time.

---

## 14. Multitask Psychometric Prediction

### Continuous Trait Head

Produces five continuous psychometric estimates:

* Engagement
* Collaboration Quality
* Socio-Emotional Presence
* Cognitive Participation
* Participation Balance

### Ordinal Head

Predicts three ordered levels for every trait:

* Low
* Moderate
* High

### Auxiliary Outcome Head

Predicts a collaborative-performance-related outcome.

### Uncertainty Head

Predicts heteroscedastic uncertainty for the continuous trait estimates.

The model therefore produces both:

```text
Predicted psychometric score
+
Predictive uncertainty
```

---

## 15. Temporal-Consistency Regularization

Window-level psychometric predictions are constrained to avoid implausible abrupt fluctuations.

The stability loss penalizes excessive change between consecutive temporal windows.

Conceptually:

```text
L_stability = mean(||T_hat(w) - T_hat(w-1)||²)
```

The default selected stability coefficient is:

```text
λs = 0.10
```

The implementation also supports sensitivity analysis using:

```text
0
0.01
0.05
0.10
0.20
0.50
```

---

## 16. Multitask Loss

The default multitask objective combines:

* Regression loss
* Ordinal classification loss
* Auxiliary prediction loss
* Temporal stability loss

Default weights:

```text
λreg = 1.00
λord = 0.50
λaux = 0.25
λs   = 0.10
```

For uncertainty-aware regression, the implementation supports Gaussian negative log-likelihood using predicted heteroscedastic variance.

---

## 17. Contrastive Pretraining

The framework optionally performs self-supervised contrastive pretraining before supervised psychometric learning.

Positive examples are generated from temporally or contextually related windows from the same collaborative session.

Negative examples are sampled from unrelated windows or sessions.

The contrastive objective encourages semantically and behaviourally related interaction windows to occupy nearby regions in the latent space.

---

## 18. Cohort-Wise Cross-Validation

Three simulated cohorts are used.

Evaluation follows a leave-one-cohort-out design.

### Fold 1

```text
Train: Cohorts 2 + 3
Test : Cohort 1
```

### Fold 2

```text
Train: Cohorts 1 + 3
Test : Cohort 2
```

### Fold 3

```text
Train: Cohorts 1 + 2
Test : Cohort 3
```

Experiments are repeated using:

```text
42
123
2026
```

as independent random seeds.

This produces repeated fold-level estimates for variability and statistical analysis.

---

## 19. Modality Configurations

### Text Only

```text
Text
```

### Core Multimodal

```text
Text + Audio + Behavioural Logs
```

### Full Multimodal

```text
Text + Audio + Behavioural Logs + Visual
```

These configurations can be used to quantify the contribution of increasing multimodal evidence.

---

## 20. Baseline Models

### Early Concatenation

Modality embeddings are concatenated before the prediction network.

### Late Fusion

Independent modality predictions are combined at the output level.

### Multimodal Transformer Fusion

Modality representations are treated as tokens and jointly processed using self-attention.

### Recurrent Fusion

Aligned multimodal features are fused and processed using a recurrent architecture.

### Conventional Multimodal Fusion

A simpler multimodal architecture without the proposed measurement-aware stability, uncertainty, and explainability mechanisms.

All baseline comparisons should use identical train/test splits and evaluation conditions.

---

## 21. Evaluation Metrics

### Continuous Trait Prediction

* MAE
* RMSE
* R²
* Pearson correlation

### Ordinal Trait Prediction

* Accuracy
* Macro-F1
* Weighted Cohen's Kappa

### Auxiliary Outcome Prediction

* Precision
* Recall
* F1-score

---

## 22. Uncertainty and Calibration

Predictive uncertainty is evaluated independently from predictive accuracy.

The framework supports:

* Gaussian NLL
* Expected Calibration Error
* Regression reliability analysis
* 90% prediction-interval coverage
* 95% prediction-interval coverage
* Correlation between uncertainty and absolute prediction error

A useful uncertainty estimator should assign larger uncertainty to predictions that exhibit larger realized errors.

---

## 23. Explainability

The implementation uses **Integrated Gradients** to identify interaction windows that contribute strongly to individual psychometric predictions.

For each prediction, the explainability module can provide:

* Predicted trait
* Predicted score
* Prediction uncertainty
* Important interaction windows
* Important textual evidence
* Modality contribution weights
* Trait trajectory

This supports evidence-linked interpretation rather than presenting psychometric predictions as unexplained model outputs.

---

## 24. Modality Contribution Analysis

The learned gating mechanism produces modality contribution weights.

Example:

| Modality         | Contribution |
| ---------------- | -----------: |
| Text             |         0.42 |
| Audio            |         0.18 |
| Behavioural Logs |         0.31 |
| Visual           |         0.09 |

These scores represent learned modality contribution within the fusion mechanism.

They should be interpreted together with local attribution methods rather than as standalone causal explanations.

---

## 25. Adaptive Intervention Engine

The implementation includes a rule-based intervention layer that translates predicted psychometric states into interpretable collaborative-learning support.

Examples include:

```text
High uncertainty
→ Instructor review recommended

Low participation balance
→ Encourage contributions from less-active group members

Low engagement
→ Generate an engagement-oriented collaborative prompt

Low collaboration quality
→ Encourage clarification, coordination, or consensus-building

Persistent socio-emotional decline
→ Flag the interaction for supportive instructional attention
```

The intervention engine is intentionally deterministic and transparent.

---

## 26. Ablation Analysis

The implementation supports systematic removal or replacement of individual components.

Examples include:

* Full Framework
* Without Cross-Modal Attention
* Without Learned Gating
* Without Audio
* Without Behavioural Logs
* Without Visual Features
* Without Temporal Stability
* Without Calibration
* Without Uncertainty Estimation

Fusion alternatives include:

* Concatenation
* Mean/Average Fusion
* Cross-Modal Attention Fusion

Ablation experiments help determine which components contribute meaningfully to performance.

---

## 27. Missing-Modality Robustness

The proposed architecture explicitly supports incomplete multimodal observations.

Unavailable modalities are masked before attention/gating operations.

Experiments can evaluate performance under increasing modality loss to determine whether the proposed fusion mechanism degrades gracefully when individual information streams are unavailable.

---

## 28. Synthetic Label-Leakage Diagnostics

Since the study uses simulated data, the repository includes diagnostics designed to verify that high predictive performance is not caused by direct target leakage.

### Univariate Screening

Examines relationships between individual observable features and reference labels.

### Label Permutation

Reference labels are randomly shuffled while retaining the original feature data.

A valid implementation should show substantial performance collapse after permutation.

### Single-Feature Experiments

Individual features are evaluated separately to determine whether one feature can reconstruct target labels.

### Single-Modality Experiments

Text-only, audio-only, log-only, and visual-only configurations are compared against full multimodal modelling.

---

## 29. Psychometric Measurement Evaluation

The repository includes support for measurement-oriented analysis beyond standard predictive metrics.

The analyses include:

* Composite Reliability
* Average Variance Extracted
* Construct Correlations
* HTMT
* Convergent Validity
* Discriminant Validity
* Cross-Cohort Measurement Analysis

Where strict multi-group confirmatory factor analysis assumptions cannot be satisfied directly by the computational representation, proxy analyses are explicitly labelled rather than presented as formal psychometric invariance tests.

---

## 30. Statistical Testing

Repeated experimental results are summarized using:

* Mean
* Standard deviation
* 95% confidence interval

For pairwise comparisons:

1. Paired differences are calculated.
2. Shapiro-Wilk normality testing is performed.
3. A paired t-test is used when the normality assumption is satisfied.
4. Otherwise, the Wilcoxon signed-rank test is applied.

Effect sizes include:

* Cohen's dz
* Rank-biserial correlation

Statistical significance is evaluated at:

```text
α = 0.05
```

---

## 31. Output Files

Depending on the selected experimental mode, the implementation may generate:

```text
outputs/
│
├── datasets/
│   ├── synthetic_sessions.*
│   └── generation_metadata.*
│
├── models/
│   └── trained_model_checkpoints.*
│
├── predictions/
│   ├── fold_predictions.*
│   └── uncertainty_predictions.*
│
├── metrics/
│   ├── regression_metrics.*
│   ├── ordinal_metrics.*
│   ├── calibration_metrics.*
│   └── statistical_results.*
│
├── tables/
│   ├── baseline_comparison.*
│   ├── ablation_results.*
│   └── psychometric_validation.*
│
└── figures/
    ├── trait_performance.*
    ├── ablation_analysis.*
    ├── calibration.*
    ├── trait_trajectories.*
    ├── modality_contributions.*
    └── sensitivity_analysis.*
```

---

## 32. Reproducibility

The main random seeds are:

```python
SEEDS = [42, 123, 2026]
```

Reproducibility controls are applied to:

* Python random number generation
* NumPy
* PyTorch
* Synthetic session generation
* Train/validation splits
* Experiment repetitions

GPU-level determinism may still depend on the CUDA and PyTorch versions used.

---

## 33. Recommended Hardware

### Quick Mode

```text
CPU
8–16 GB RAM
```

### Full Experimental Mode

```text
NVIDIA GPU
16+ GB system RAM
CUDA-enabled PyTorch
```

Google Colab GPU runtime can also be used.

---

## 34. Research Use

This repository is intended for:

* Multimodal learning analytics research
* Collaborative-learning analytics
* Automated psychometric modelling
* Learner-state modelling
* Educational NLP
* Multimodal fusion research
* Explainable AI in education
* Uncertainty-aware educational assessment
* Simulation-based methodological evaluation

---

## 35. Ethical Scope

The default implementation operates entirely on simulated collaborative-learning sessions.

It does not require:

* Human participants
* Personally identifiable information
* Institutional learner records
* Real classroom recordings
* Raw facial images
* Private audio recordings

The current implementation is therefore intended primarily for methodological validation.

Real-world educational deployment would require additional consideration of:

* Informed consent
* Privacy
* Data minimization
* Institutional approval
* Bias and fairness
* Demographic validation
* Cultural validity
* Teacher oversight
* Human-in-the-loop decision making

---

## 36. Limitations

The current framework is validated primarily using synthetically generated multimodal interaction data.

Consequently, strong performance under the simulation environment should not be interpreted as established effectiveness in real classrooms.

Important future validation directions include:

* Authentic collaborative-learning datasets
* Longitudinal learner studies
* Cross-cultural validation
* Institutional deployment
* Demographic fairness analysis
* Privacy-preserving multimodal processing
* Teacher-in-the-loop validation
* Comparison with established psychometric instruments
* Real-time intervention evaluation

---

## 37. Future Extensions

Potential extensions include:

* Real multimodal classroom datasets
* Speech foundation models
* Large language model-based discourse encoding
* Multimodal foundation models
* Graph-based collaboration modelling
* Dynamic learner-state graphs
* Federated multimodal learning
* Privacy-preserving representation learning
* Differential privacy
* Multimodal missing-data imputation
* Bayesian psychometric modelling
* Item-response-theory integration
* Longitudinal psychometric trajectories
* Causal intervention evaluation
* Teacher-facing analytics dashboards

---

## 38. Citation

If you use this implementation in academic work, please cite the associated research paper once its publication information becomes available.

### Suggested Temporary Repository Citation

```text
Multimodal NLP Framework for Automated Psychometric Assessment in Collaborative Learning.

Python/PyTorch research implementation for simulation-based multimodal
psychometric assessment, cross-modal fusion, uncertainty-aware prediction,
explainability, and adaptive intervention.
```

A formal BibTeX entry can be added after publication.

---

## 39. License

Add the preferred repository license before public release.

For academic and research-code repositories, commonly used options include:

* MIT License
* Apache License 2.0
* BSD 3-Clause License

Select the license according to the intended reuse and redistribution conditions.

---

## 40. Contact

For research collaboration, implementation questions, reproducibility issues, or academic use, please contact the repository maintainer through the GitHub repository.

---

## Disclaimer

> **Important:** This software is a research prototype intended for methodological and experimental purposes.

Psychometric predictions produced by the framework should **not** be used as standalone clinical, psychological, disciplinary, or high-stakes educational decisions without appropriate human oversight and empirical validation on real populations.
