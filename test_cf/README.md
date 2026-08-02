# test_cf — Crazyflie LOTF Test Bench

This folder contains the Crazyflie-specific part of the project: LOTF policy training, evaluation in Genesis, simulated fine-tuning, real flight, and online fine-tuning on the real drone. For the project overview, methodology and results, see the [root README](../README.md).

## Useful commands

All commands below are run from the repository root unless stated otherwise.

### 0. Environments

Virtual environments are not versioned. Only the lock/config files needed to recreate them are kept.

```bash
# Root JAX/LOTF environment: JAX pretraining, JAX fine-tuning, notebooks.
uv sync

# Crazyflie radio environment: cflib + real-time control.
cd test_cf/RL-real
uv venv .venv --python 3.11
uv pip install torch numpy scipy matplotlib pandas cflib pynput pyyaml
cd ../..

# Local Genesis environment, for 3D simulation.
# Install Genesis for your machine/GPU, then activate it before Genesis commands.
uv venv genesis_venv --python 3.11
genesis_venv/bin/python -m pip install genesis-world torch numpy scipy
source genesis_venv/bin/activate
```

### 1. Policy pretraining

Recommended version, aligned with the JAX engine used afterward for fine-tuning:

```bash
cd test_cf
../.venv/bin/python core/pretrain_lotf_jax.py --preset base27 --out models/model_pretrain.pt
```

Useful variants to probe the sim/real mass gap:

```bash
../.venv/bin/python core/pretrain_lotf_jax.py --preset mass33 --out models/model_33g.pt
../.venv/bin/python core/pretrain_lotf_jax.py --preset res27  --out models/model_res33.pt
```

Older standalone torch version, kept in `RL-real/`:

```bash
cd test_cf/RL-real
.venv/bin/python pretrain_lotf.py --epochs 300 --out ../models/model_pretrain_torch.pt
```

### 2. Flying in Genesis

Simple evaluation of a checkpoint:

```bash
cd test_cf
source ../genesis_venv/bin/activate
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --steps 15000
```

Without a graphical viewer:

```bash
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --steps 15000 --no-viewer
```

Recording a rollout usable by fine-tuning:

```bash
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt \
    --log-lotf measurements/genesis/rollout.npz --no-viewer
```

### 3. Fine-tuning in Genesis

Offline fine-tuning from a Genesis rollout:

```bash
cd test_cf
../.venv/bin/python core/finetune_lotf_jax.py \
    --base models/model_pretrain.pt \
    --log measurements/genesis/rollout.npz \
    --out models/model_ft_jax.pt
```

Learning-on-the-fly demo in Genesis, with hot-swap mid-flight:

```bash
cd test_cf
source ../genesis_venv/bin/activate
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt \
    --online-finetune --jax-python ../.venv/bin/python --steps 15000
```

Keys during the Genesis demo: `f` then Enter starts fine-tuning, `m` changes the simulated mass, `q` then Enter quits.

### 4. Real flight

Clear area, Flow deck detected, fail-safe ready. Radio control uses the `test_cf/RL-real/.venv` environment, JAX fine-tuning uses `../.venv`.

```bash
cd test_cf
RL-real/.venv/bin/python core/lotf_real_eval.py \
    --model models/model_pretrain.pt \
    --session measurements/real/real_run
```

Generated outputs:

```text
measurements/real/real_run_lotf.npz
measurements/real/real_run.csv
measurements/real/real_run_traj.png
measurements/real/real_run_thrust.png
```

Keyboard controls in real flight: hold `ENTER` as a dead-man switch, release `ENTER` = immediate kill, `SPACE` = controlled landing.

### 5. Real-world fine-tuning

Full run with online fine-tuning:

```bash
cd test_cf
RL-real/.venv/bin/python core/lotf_real_eval.py \
    --model models/model_pretrain.pt \
    --online-finetune \
    --jax-python ../.venv/bin/python \
    --session measurements/real/real_run
```

Automatic triggering is configured in `core/lotf_real_eval.py`: the RL window starts after `WARMUP_END_S = 5.0 s`, then fine-tuning triggers at `AUTO_FT_RL_SEC = 15.0 s` after that RL start. In the CSVs, `finetune_trigger` is therefore the actual trigger time, not a fixed "10 s before hot-swap" rule.

### 6. Plots and diagnostics

Main plotting notebook:

```bash
cd test_cf
../.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/courbes.ipynb \
    --output /tmp/courbes_executed.ipynb
```

Example figures live in `assets/figures/` (and GIFs in `assets/gifs/`). Raw measurements stay in `measurements/` and should not be mixed with presentation assets.

Quick tests/diagnostics:

```bash
cd test_cf
../.venv/bin/python tests/verify_vs_gazebo.py
../.venv/bin/python tests/test_finetune_unit.py
python tests/test_genesis_finetune.py --ckpt models/model_pretrain.pt --no-viewer
```

## Directory layout

```text
test_cf/
├── README.md
├── configs/
│   └── lotf_config.yaml          # LOTF/fine-tuning/Genesis hyperparameters
├── core/                         # main reusable code
│   ├── lotf_genesis_eval.py      # Genesis simulation entry point
│   ├── lotf_real_eval.py         # real Crazyflie entry point
│   ├── lotf_online.py            # shared learning-on-the-fly core
│   ├── lotf_genesis_env.py       # Genesis boundary
│   ├── lotf_real_env.py          # cflib radio boundary
│   ├── pretrain_lotf_jax.py      # JAX pretraining
│   └── finetune_lotf_jax.py      # offline JAX fine-tuning
├── RL-real/                      # legacy standalone Crazyflie port + shared primitives
│   ├── RL_policy.py              # torch network loaded by Genesis and the real drone
│   ├── cf_params.py              # authoritative physical constants
│   ├── drone_state.py            # 27-dim observation + action FIFO
│   └── drone_controller.py       # historical radio control logic
├── models/                       # saved .pt checkpoints
├── measurements/                 # raw logs/CSV/NPZ
│   ├── genesis/
│   └── real/
├── assets/                       # presentation assets, no raw measurements
│   ├── figures/
│   ├── gifs/
│   └── videos/
├── notebooks/                    # plotting/exploration notebooks, kept clean
├── tests/                        # unit tests, diagnostics, comparisons
├── scripts/                      # non-core utility scripts
├── docs/                         # article and end-of-project notes
└── legacy/                       # historical backups not used by the pipeline
```

`core/` is the best place to start reading to understand the current pipeline. `RL-real/` remains necessary because `core/` reuses its stable building blocks (`cf_params`, `DroneState`, `RLPolicy`) so that Genesis and the real drone share the same observation/action conventions.

## Configuration

The main configuration is `configs/lotf_config.yaml`. It is loaded by `core/lotf_config.py` and drives:

- `residual_fit`: learning the residual dynamics model.
- `bptt`: policy adaptation via backpropagation through time.
- `hovering_env`: reward and environment randomization for LOTF.
- `online`: data window, epoch count, number of hot-swaps, anti-divergence guard.
- `jax`: XLA compilation cache, generated in `.jax_cache/` and git-ignored.
- `genesis_bridge`: Genesis bridge calibration and the mass cycle for the `m` key.

Platform physical constants are not edited in the YAML: the source of truth remains `RL-real/cf_params.py` for nominal mass, timestep, action bounds, hover target, and observation structure.

## Assets and hygiene

`assets/` contains only presentable examples: plots, images, GIFs and videos. `measurements/` contains raw flight/simulation data. `models/` contains useful checkpoints. Caches (`.jax_cache/`, `__pycache__/`, `measurements/online_ft/`) and virtual environments (`.venv/`) are regenerable and git-ignored.
