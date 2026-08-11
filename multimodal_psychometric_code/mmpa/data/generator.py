from __future__ import annotations
import math, random
from pathlib import Path
import numpy as np
import torch
from scipy.stats import truncnorm
from mmpa.utils import seed_everything, ensure_dir

TEXT_TEMPLATES = {
    'agreement': ['I agree with that approach.', 'That interpretation makes sense.', 'Yes, this is consistent with our result.'],
    'clarification': ['Could you explain the second step?', 'Can we clarify this assumption?', 'What exactly do you mean here?'],
    'question': ['What should we test next?', 'Why does this result occur?', 'Can someone justify this choice?'],
    'disagreement': ['I see the point differently.', 'I am not convinced by that conclusion.', 'That may conflict with the earlier evidence.'],
    'repair': ['Let me correct what I said.', 'We should revise that step.', 'I think we need to repair the previous reasoning.'],
    'summary': ['Let me summarize our progress.', 'So far, these are the main findings.', 'Our current conclusion is as follows.'],
    'task': ['I will work on the next calculation.', 'Let us compare the two alternatives.', 'We should complete the current task first.'],
    'support': ['That is a useful contribution.', 'Good point; we can build on it.', 'Your explanation helps the group.'],
    'reflection': ['We may need to reconsider our strategy.', 'What did we learn from the previous attempt?', 'Let us reflect on why this worked.'],
    'coordination': ['You take the first part and I will take the second.', 'Let us divide the remaining tasks.', 'We should synchronize our contributions.'],
}
EVENTS = list(TEXT_TEMPLATES)


def _clip(x):
    return np.clip(x, 0.0, 1.0)


def _truncnorm(rng, mu, sd, size):
    a, b = (0-mu)/sd, (1-mu)/sd
    # scipy truncnorm random_state accepts numpy Generator
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size, random_state=rng)


def _event_probs(t):
    e, c, s, g, b = t
    raw = np.array([
        .20 + .55*c,                 # agreement
        .15 + .35*c + .20*g,        # clarification
        .15 + .55*g,                 # question
        .15 + .35*(1-c) + .10*(1-s),# disagreement
        .10 + .30*c,                 # repair
        .10 + .40*g,                 # summary
        .15 + .45*e,                 # task
        .10 + .50*s,                 # support
        .10 + .35*g + .10*s,        # reflection
        .10 + .45*c + .15*b,        # coordination
    ])
    raw += 0.02
    return raw / raw.sum()


def _generate_text(rng, trait):
    probs = _event_probs(trait)
    n_events = int(rng.integers(1, 4))
    chosen = rng.choice(EVENTS, size=n_events, replace=True, p=probs)
    parts = []
    for ev in chosen:
        phrase = random.choice(TEXT_TEMPLATES[ev])
        # light lexical perturbation without deterministic label tokens
        if rng.random() < .25:
            phrase = phrase.replace('Let us', "Let's")
        parts.append(phrase)
    return ' '.join(parts)


def _generate_audio(rng, t, noise):
    e,c,s,g,b = t
    base = np.array([
        .25 + .60*e,               # speaking rate
        .70 - .45*e + .10*(1-s),  # pause duration (normalized inverse activity)
        .25 + .45*e + .10*g,      # turn duration
        .20 + .55*s,              # pitch variation
        .20 + .50*s,              # intensity variation
        .35 + .35*(1-b),          # overlap frequency
        .20 + .55*b,              # speaker alternation regularity
        .25 + .45*e + .15*s,      # voice activity ratio
    ])
    return _clip(base + rng.normal(0, noise, size=8))


def _generate_logs(rng, t, noise):
    e,c,s,g,b = t
    # normalized stochastic descriptors, each with its own process noise
    lam_msg = 1 + 6*(.5*e + .5*g)
    lam_edit = .5 + 4*g
    lam_turn = 1 + 5*(.5*c + .5*b)
    msg = min(rng.poisson(lam_msg)/10.0, 1.0)
    edits = min(rng.poisson(lam_edit)/8.0, 1.0)
    turns = min(rng.poisson(lam_turn)/9.0, 1.0)
    latency = _clip(rng.lognormal(mean=-.3-.9*c, sigma=.35)/2.5)
    contribution = _clip(.25 + .55*b + rng.normal(0, noise))
    alternation = _clip(.20 + .60*(.5*c+.5*b) + rng.normal(0, noise))
    coordination = _clip(.15 + .65*c + rng.normal(0, noise))
    dominance = _clip(.75 - .60*b + rng.normal(0, noise))
    return _clip(np.array([msg,edits,turns,latency,contribution,alternation,coordination,dominance]))


def _generate_visual(rng, t, noise):
    e,c,s,g,b = t
    base = np.array([
        .20 + .55*s,              # positive affect
        .55 - .40*s,              # negative affect
        .45 - .20*abs(s-.5),      # neutral affect
        .20 + .65*e,              # attention
        .25 + .55*e,              # gaze stability
        .20 + .50*s + .10*e,     # facial activation
    ])
    return _clip(base + rng.normal(0, noise, size=6))


def generate_sessions(cfg, seed=42):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    sessions = []
    per_cohort = cfg.n_sessions // cfg.n_cohorts
    for sid in range(cfg.n_sessions):
        cohort = min(sid // max(1, per_cohort), cfg.n_cohorts-1)
        group_size = int(rng.integers(cfg.group_size_min, cfg.group_size_max+1))
        duration_min = int(rng.integers(cfg.duration_min, cfg.duration_max+1))
        n_windows = max(2, int(duration_min*60/cfg.window_seconds))
        init = _truncnorm(rng, cfg.latent_mu, cfg.latent_sd, cfg.n_traits)
        latent = np.zeros((n_windows, cfg.n_traits), dtype=np.float32)
        latent[0] = init
        for w in range(1, n_windows):
            latent[w] = _clip(cfg.temporal_persistence*latent[w-1] +
                              (1-cfg.temporal_persistence)*init +
                              rng.normal(0, cfg.process_noise_sd, cfg.n_traits))
        ref_noise = rng.normal(0, 0.025, size=latent.shape)
        y_window = _clip(latent + ref_noise).astype(np.float32)
        y_session = y_window.mean(axis=0).astype(np.float32)
        y_ord = np.digitize(y_session, bins=[0.33,0.67], right=False).astype(np.int64)
        # binary collaborative outcome from separate noisy multivariate function
        aux_latent = .28*y_session[0] + .30*y_session[1] + .15*y_session[2] + .17*y_session[3] + .10*y_session[4]
        aux_score = float(_clip(aux_latent + rng.normal(0,.08)))
        aux = np.float32(aux_score >= .52)

        text=[]; audio=[]; logs=[]; visual=[]; masks=[]
        noise = float(rng.uniform(cfg.obs_noise_min, cfg.obs_noise_max))
        for w in range(n_windows):
            t = latent[w]
            text.append(_generate_text(rng, t))
            audio.append(_generate_audio(rng,t,noise))
            logs.append(_generate_logs(rng,t,noise))
            visual.append(_generate_visual(rng,t,noise))
            mask = np.array([
                1.0,
                float(rng.random() >= cfg.missing_audio),
                float(rng.random() >= cfg.missing_logs),
                float(rng.random() >= cfg.missing_visual),
            ], dtype=np.float32)
            masks.append(mask)
        audio=np.asarray(audio,np.float32); logs=np.asarray(logs,np.float32); visual=np.asarray(visual,np.float32); masks=np.asarray(masks,np.float32)
        audio[masks[:,1]==0]=0; logs[masks[:,2]==0]=0; visual[masks[:,3]==0]=0
        sessions.append({
            'session_id': sid, 'cohort_id': cohort, 'group_size': group_size,
            'duration_min': duration_min, 'n_windows': n_windows,
            'text': text, 'audio': audio, 'logs': logs, 'visual': visual, 'mask': masks,
            'latent': latent, 'y_window': y_window, 'y_session': y_session,
            'y_ord': y_ord, 'aux_score': np.float32(aux_score), 'y_aux': aux,
            'noise_level': noise,
        })
    return sessions


def save_sessions(sessions, path):
    ensure_dir(Path(path).parent)
    torch.save(sessions, path)


def load_sessions(path):
    return torch.load(path, weights_only=False)
