# Manuscript-to-code alignment

- Synthetic protocol, latent traits, noise, missingness and 3 cohorts: `mmpa/data/generator.py`
- Temporal synchronization/session padding: `mmpa/data/dataset.py`
- Modality-specific text/audio/log/visual encoders: `mmpa/models/encoders.py`
- Window-level cross-modal attention + availability-aware gating: `mmpa/models/fusion.py`
- Session temporal aggregation and multi-task heads: `mmpa/models/model.py`
- Continuous regression + heteroscedastic uncertainty + ordinal + auxiliary + stability objective: `mmpa/training/losses.py`
- Contrastive pretraining, supervised training and cohort-wise evaluation: `mmpa/training/trainer.py`, `mmpa/experiments/run_experiments.py`
- Five baseline fusion architectures: `mmpa/models/baselines.py`
- Calibration and uncertainty quality: `mmpa/evaluation/uncertainty.py`
- MAE/RMSE/R2, ordinal metrics, auxiliary classification metrics: `mmpa/evaluation/metrics.py`
- CR/AVE/HTMT and cohort measurement diagnostic: `mmpa/evaluation/psychometrics.py`
- Leakage diagnostics: `mmpa/evaluation/leakage.py`
- Shapiro-Wilk, paired t/Wilcoxon and effect sizes: `mmpa/evaluation/statistics.py`
- Integrated Gradients salient windows: `mmpa/explainability/integrated_gradients.py`
- Uncertainty-aware rule intervention: `mmpa/explainability/intervention.py`
- End-to-end orchestration, baselines, ablations, sensitivity, figures and exports: `mmpa/experiments/run_experiments.py`
