# Multimodal NLP Framework for Automated Psychometric Assessment in Collaborative Learning

This repository is a complete, self-contained Python/PyTorch implementation of the manuscript methodology. It covers synthetic multimodal data generation, modality-specific encoders, cross-modal attention, availability-aware gating, temporal aggregation, multi-task psychometric prediction, heteroscedastic uncertainty, ordinal calibration, temporal stability regularization, Integrated Gradients, adaptive intervention rules, baseline comparisons, ablations, leakage diagnostics, psychometric construct diagnostics, statistical tests, and publication-ready output files.

## Important design decisions

- The paper's **fixed 5-second window** is used in paper mode.
- The synthetic study generates **acoustic descriptors**, not WAV files; therefore the audio branch uses the generated prosodic/acoustic feature vectors directly. This keeps the implementation faithful to the experimental protocol while avoiding invented raw audio.
- The text branch is a **local Transformer encoder** (token embedding + TransformerEncoder) and requires no Hugging Face/model download, making the project reproducible offline after dependencies are installed.
- Cross-modal attention is applied **within each synchronized window**. Learned gating is applied after attention to determine modality contribution. Temporal dependencies are then modeled with a BiGRU + attention pooling.
- Continuous traits use a heteroscedastic Gaussian NLL head (mean + log variance). Ordinal predictions use 3 ordered levels per trait; the auxiliary task is binary collaborative success so Precision/Recall/F1 are well defined.
- The default `quick` mode is a smoke test. `paper` mode restores the manuscript-scale 120 sessions, 3 cohorts, 15–30 min sessions, 50 epochs, three seeds, three held-out-cohort folds, five baselines, and stability sensitivity.

## Project structure

```text
mmpa/
  config.py
  utils.py
  data/
    generator.py
    dataset.py
  models/
    encoders.py
    fusion.py
    model.py
    baselines.py
  training/
    losses.py
    trainer.py
  evaluation/
    metrics.py
    uncertainty.py
    statistics.py
    leakage.py
    psychometrics.py
  explainability/
    integrated_gradients.py
    intervention.py
  experiments/
    run_experiments.py
run_all.py
requirements.txt
notebooks/MultimodalPsychometric_Colab.ipynb
```

## Installation

### Google Colab

Upload/extract the ZIP, open a terminal/cell in the project directory, then run:

```bash
pip install -r requirements.txt
```

### Local Python

Python 3.10+ is recommended.

```bash
python -m venv .venv
# activate the environment, then
pip install -r requirements.txt
```

## Run a complete smoke test

```bash
python run_all.py --mode quick --output-dir outputs_quick
```

This runs the proposed model plus all five baselines on a reduced but structurally identical dataset and verifies that the full pipeline executes.

## Run manuscript-scale experiments

```bash
python run_all.py --mode paper --output-dir outputs_paper
```

Paper mode can take substantial GPU time because it performs repeated cohort-held-out training across multiple models, seeds and folds.

## Optional quick stability sensitivity

```bash
python run_all.py --mode quick --output-dir outputs_quick_sensitivity --sensitivity
```

## Main paper-mode configuration

- Sessions: 120
- Cohorts: 3 × 40
- Learners/group: 3–5
- Session duration: 15–30 min
- Temporal window: 5 s
- Traits: 5
- Trait range: [0, 1]
- Initial latent state: truncated Normal(0.50, 0.15²)
- Temporal persistence: 0.85
- Process-noise SD: 0.03
- Observable noise: 5–15%
- Missingness: text 0%, audio 5%, logs 8%, visual 20%
- Seeds: 42, 123, 2026
- Loss weights: regression 1.0, ordinal 0.5, auxiliary 0.25, stability 0.10
- Epochs: 50
- Batch size: 16
- Optimizer: AdamW (Adam-family implementation) with learning rate 1e-4

## Outputs

The runner creates:

```text
outputs_*/
  config.json
  synthetic_sessions.pt
  models/
  predictions/
  tables/
  figures/
  explainability_case.json
```

Important tables include:

- `experiment_summary.csv`
- per-trait metric tables
- calibrated predictions
- `modality_ablation.csv`
- `stability_sensitivity.csv` (paper mode or `--sensitivity`)
- `univariate_leakage_screen.csv`
- `permutation_linear_diagnostic.json`
- `construct_quality.csv`
- `htmt_matrix.csv`
- `measurement_invariance_proxy.csv`
- `paired_statistical_tests.csv`

## Psychometric validation note

CR, AVE, HTMT and construct correlations are computed directly from the synthetic observable indicators. The included `measurement_invariance_proxy.csv` is deliberately named a **proxy**: it reports the same five-construct indicator structure plus cross-cohort loading/intercept drift without pretending to be a full constrained multi-group CFA package. This is preferable to silently inventing CFI/RMSEA from an invalid approximation. If the final manuscript requires strict configural/metric/scalar multi-group CFA with ΔCFI and ΔRMSEA, perform that confirmatory analysis in a dedicated SEM package (e.g., lavaan/Mplus) on the exported indicator table.

## Reproducibility

All synthetic-data and model runs are seeded. Reference labels are generated through a branch independent of observable features, preserving the intended causal structure `T -> X` and `T -> Y` with no direct `Y -> X` construction.

## Verified smoke tests

The supplied project was syntax-checked and executed end-to-end in quick mode for:

1. proposed model only,
2. proposed model + all five baselines,
3. proposed model + stability sensitivity.

Warnings about weighted kappa can occur in the tiny quick dataset when a held-out fold contains only one ordinal label; this does not indicate a runtime failure and is not expected at manuscript scale.
