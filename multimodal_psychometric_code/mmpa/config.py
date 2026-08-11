from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class Config:
    # Paper-locked synthetic protocol
    n_sessions: int = 120
    n_cohorts: int = 3
    group_size_min: int = 3
    group_size_max: int = 5
    duration_min: int = 15
    duration_max: int = 30
    window_seconds: int = 5
    n_traits: int = 5
    trait_names: List[str] = field(default_factory=lambda: [
        'engagement','collaboration_quality','socio_emotional_presence',
        'cognitive_participation','participation_balance'])
    seeds: List[int] = field(default_factory=lambda: [42,123,2026])
    latent_mu: float = 0.50
    latent_sd: float = 0.15
    temporal_persistence: float = 0.85
    process_noise_sd: float = 0.03
    obs_noise_min: float = 0.05
    obs_noise_max: float = 0.15
    missing_text: float = 0.00
    missing_audio: float = 0.05
    missing_logs: float = 0.08
    missing_visual: float = 0.20

    # Model
    vocab_size: int = 4096
    max_tokens: int = 24
    text_embed_dim: int = 96
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.10
    temporal_hidden: int = 96
    audio_dim: int = 8
    log_dim: int = 8
    visual_dim: int = 6

    # Training
    batch_size: int = 16
    epochs: int = 50
    pretrain_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    grad_clip: float = 1.0
    lambda_reg: float = 1.0
    lambda_ord: float = 0.50
    lambda_aux: float = 0.25
    lambda_stability: float = 0.10
    contrastive_temperature: float = 0.07

    # Execution
    device: str = 'auto'
    num_workers: int = 0
    output_dir: str = 'outputs'
    mode: str = 'paper'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def quick(cls):
        """Small deterministic smoke-test configuration; paper mode remains available."""
        c = cls()
        c.n_sessions = 12
        c.duration_min = 1
        c.duration_max = 2
        c.batch_size = 4
        c.epochs = 2
        c.pretrain_epochs = 1
        c.early_stopping_patience = 2
        c.mode = 'quick'
        return c
