# Advanced Coding and Machine Learning — Coursework

Third year BSc Physics coursework from the Advanced Coding and Machine Learning module at the University of Bristol (2025–2026).

---

## Contents

| Assignment | Topic | Key Methods |
|---|---|---|
| [Assignment 1 — Monte Carlo Particle Simulation](#assignment-1--monte-carlo-particle-physics-simulation) | Particle physics detector simulation | Monte Carlo, NumPy vectorisation, statistical validation |
| [Assignment 2 — Solar Power Forecasting](#assignment-2--solar-power-forecasting-with-deep-learning) | Time-series energy prediction with deep learning | LSTM, GRU, CNN-fusion, Optuna hyperparameter tuning |

---

## Assignment 1 — Monte Carlo Particle Physics Simulation

**File:** `Project_1_-_Option_B.ipynb`

### Overview
A fully vectorised Monte Carlo simulation of a particle beam decay experiment. A beam of 10⁶ particles travels along the z-axis, each decaying into a daughter particle after a randomly drawn lifetime. The simulation tracks daughter particles across four square detector stations and analyses hit distributions and detection rates.

### Physics Setup
- Beam velocity drawn from a normal distribution: μ = 2000 ms⁻¹, σ = 50 ms⁻¹
- Decay time drawn from an exponential distribution: mean lifetime τ = 2.5 ms
- Daughter particles emitted isotropically from the decay point
- Four tracking stations at z = 30, 35, 40, 45 m (5m × 5m each)
- Detector resolution: ±0.01 m Gaussian measurement uncertainty

### Key Implementation Details
- Isotropic emission validated with a Kolmogorov–Smirnov test on cos(θ) at the 5% significance level
- Fully vectorised using NumPy arrays (no particle-by-particle loops in core computation)
- Hit positions recorded with realistic Gaussian errors and finite spatial resolution
- Expected hit count derived analytically via solid-angle integration and compared against simulation output

### Methods & Technologies
- Python, NumPy, Matplotlib, SciPy, Pandas
- Monte Carlo sampling, KS statistical testing, solid angle calculations

---

## Assignment 2 — Solar Power Forecasting with Deep Learning

**File:** `Assignment_2_code.ipynb`

### Overview
A deep learning pipeline for forecasting solar power output from real photovoltaic site data. The project addresses both single-site and generalised multi-site prediction across two time horizons, comparing multiple neural architectures against a naive persistence baseline.

### Tasks
1. **Single-site, 15-minute horizon** — predict next 15-minute power output for a single PV site (UG Hall 7)
2. **Single-site, 24-hour horizon** — multi-step prediction 96 steps (24 hours) ahead
3. **Generalised multi-site model** — a single model trained across multiple sites that generalises to unseen holdout sites

### Models Implemented

| Model | Description |
|---|---|
| `FlexibleRNN` | Configurable LSTM or GRU for single-step prediction |
| `FlexibleMultiStepRNN` | LSTM/GRU extended to multi-step sequence output |
| `LateFusionModel` | Late-fusion architecture combining dynamic time-series with static site metadata |
| `FiLMModel` | Feature-wise Linear Modulation — conditions the RNN on static site features to aid generalisation |

### Key Engineering Choices
- **Leakage-free scaling** — MinMaxScaler fitted strictly on training data, then applied to validation and test sets
- **Cyclic time encoding** — raw timestamps converted to sine/cosine features so the model understands daily and yearly periodicity
- **Group-aware sequence building** — multi-site sequences built within individual site timelines to prevent boundary crossing between sites
- **Optuna hyperparameter tuning** — automated search over learning rate, hidden size, number of layers, and dropout
- **Early stopping** with best-model checkpointing
- **Naive persistence baseline** — all models benchmarked against a baseline that predicts the previous timestep's value

### Methods & Technologies
- Python, PyTorch, Pandas, NumPy, Scikit-learn, Optuna, Matplotlib, Seaborn
- LSTM, GRU, sequence modelling, time-series cross-validation, hyperparameter optimisation

---
