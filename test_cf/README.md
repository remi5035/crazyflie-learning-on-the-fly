# test_cf - Banc LOTF Crazyflie

Ce dossier contient la partie Crazyflie du projet : entrainement de politique LOTF, evaluation dans Genesis, fine-tuning en simulation, vol reel et fine-tuning en ligne sur le vrai drone.

## Commandes utiles

Toutes les commandes ci-dessous se lancent depuis la racine du depot, sauf mention contraire.

### 0. Environnements

Les environnements virtuels ne sont pas versionnes. Le repo conserve seulement les fichiers de lock/config utiles pour les recreer.

```bash
# Environnement JAX/LOTF racine : pretrain JAX, fine-tuning JAX, notebooks.
uv sync

# Environnement radio Crazyflie : cflib + controle temps reel.
cd test_cf/RL-real
uv venv .venv --python 3.11
uv pip install torch numpy scipy matplotlib pandas cflib pynput pyyaml
cd ../..

# Environnement Genesis local, si besoin de simulation 3D.
# Installer Genesis selon la machine/GPU, puis l'activer avant les commandes Genesis.
uv venv genesis_venv --python 3.11
genesis_venv/bin/python -m pip install genesis-world torch numpy scipy
source genesis_venv/bin/activate
```

### 1. Pre-entrainement de la politique

Version recommandee, alignee avec le moteur JAX utilise ensuite pour le fine-tuning :

```bash
cd test_cf
../.venv/bin/python core/pretrain_lotf_jax.py --preset base27 --out models/model_pretrain.pt
```

Variantes utiles pour tester l'ecart de masse sim/reel :

```bash
../.venv/bin/python core/pretrain_lotf_jax.py --preset mass33 --out models/model_33g.pt
../.venv/bin/python core/pretrain_lotf_jax.py --preset res27  --out models/model_res33.pt
```

Ancienne version torch autonome, gardee dans `RL-real/` :

```bash
cd test_cf/RL-real
.venv/bin/python pretrain_lotf.py --epochs 300 --out ../models/model_pretrain_torch.pt
```

### 2. Vol dans Genesis

Evaluation simple d'un checkpoint :

```bash
cd test_cf
source ../genesis_venv/bin/activate
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --steps 15000
```

Sans fenetre graphique :

```bash
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --steps 15000 --no-viewer
```

Enregistrer un rollout exploitable par le fine-tuning :

```bash
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt     --log-lotf measurements/genesis/rollout.npz --no-viewer
```

### 3. Fine-tuning dans Genesis

Fine-tuning offline depuis un rollout Genesis :

```bash
cd test_cf
../.venv/bin/python core/finetune_lotf_jax.py     --base models/model_pretrain.pt     --log measurements/genesis/rollout.npz     --out models/model_ft_jax.pt
```

Demo learning-on-the-fly dans Genesis, avec hot-swap pendant le vol :

```bash
cd test_cf
source ../genesis_venv/bin/activate
python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt     --online-finetune --jax-python ../.venv/bin/python --steps 15000
```

Touches pendant la demo Genesis : `f` puis Entree lance le fine-tuning, `m` change la masse simulee, `q` puis Entree quitte.

### 4. Vol en vrai

Zone degagee, Flow Deck detecte, fail-safe pret. Le controle radio utilise l'environnement `test_cf/RL-real/.venv`, le fine-tuning JAX utilise `../.venv`.

```bash
cd test_cf
RL-real/.venv/bin/python core/lotf_real_eval.py     --model models/model_pretrain.pt     --session measurements/real/real_run
```

Sorties generees :

```text
measurements/real/real_run_lotf.npz
measurements/real/real_run.csv
measurements/real/real_run_traj.png
measurements/real/real_run_thrust.png
```

Commandes clavier en vol reel : `ENTREE` maintenue = dead-man switch, relacher `ENTREE` = kill immediat, `ESPACE` = atterrissage controle.

### 5. Fine-tuning en vrai

Lancement complet avec fine-tuning en ligne :

```bash
cd test_cf
RL-real/.venv/bin/python core/lotf_real_eval.py     --model models/model_pretrain.pt     --online-finetune     --jax-python ../.venv/bin/python     --session measurements/real/real_run
```

Le declenchement automatique est configure dans `core/lotf_real_eval.py` : la fenetre RL commence apres `WARMUP_END_S = 5.0 s`, puis le fine-tuning est declenche a `AUTO_FT_RL_SEC = 15.0 s` apres ce debut RL. Dans les CSV, `finetune_trigger` correspond donc a l'instant reel de declenchement du fine-tuning, pas a une regle fixe "10 s avant hot-swap".

### 6. Courbes et diagnostics

Notebook principal de figures :

```bash
cd test_cf
../.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/courbes.ipynb     --output /tmp/courbes_executed.ipynb
```

Les figures d'exemple sont rangees dans `assets/figures/`. Les mesures brutes restent dans `measurements/` et ne doivent pas etre melangees avec les assets de presentation.

Tests/diagnostics rapides :

```bash
cd test_cf
../.venv/bin/python tests/verify_vs_gazebo.py
../.venv/bin/python tests/test_finetune_unit.py
python tests/test_genesis_finetune.py --ckpt models/model_pretrain.pt --no-viewer
```

## Organisation du repertoire

```text
test_cf/
├── README.md
├── configs/
│   └── lotf_config.yaml          # hyperparametres LOTF/fine-tuning/Genesis
├── core/                         # code principal reutilisable
│   ├── lotf_genesis_eval.py      # entree simulation Genesis
│   ├── lotf_real_eval.py         # entree vrai Crazyflie
│   ├── lotf_online.py            # coeur learning-on-the-fly commun
│   ├── lotf_genesis_env.py       # frontiere Genesis
│   ├── lotf_real_env.py          # frontiere radio cflib
│   ├── pretrain_lotf_jax.py      # pretrain JAX
│   └── finetune_lotf_jax.py      # fine-tuning offline JAX
├── RL-real/                      # ancien port Crazyflie autonome + primitives partagees
│   ├── RL_policy.py              # reseau torch charge par Genesis et le vrai drone
│   ├── cf_params.py              # constantes physiques faisant foi
│   ├── drone_state.py            # observation 27-dim + FIFO d'actions
│   └── drone_controller.py       # logique radio historique
├── models/                       # checkpoints .pt conserves
├── measurements/                 # logs/CSV/NPZ bruts
│   ├── genesis/
│   └── real/
├── assets/                       # elements de presentation, pas de mesures brutes
│   ├── figures/
│   └── videos/
├── notebooks/                    # notebooks de tracé/exploration gardes propres
├── tests/                        # tests unitaires, diagnostics, comparaisons
├── scripts/                      # scripts utilitaires non centraux
├── docs/                         # article et notes de fin de projet
└── legacy/                       # backups historiques non utilises par la pipeline
```

`core/` est la partie a lire en premier pour comprendre la pipeline actuelle. `RL-real/` reste necessaire car `core/` reutilise ses briques stables (`cf_params`, `DroneState`, `RLPolicy`) afin que Genesis et le vrai drone partagent les memes conventions d'observation/action.

## Configuration

La configuration principale est `configs/lotf_config.yaml`. Elle est chargee par `core/lotf_config.py` et pilote :

- `residual_fit` : apprentissage du modele residuel de dynamique.
- `bptt` : adaptation de la politique par backpropagation dans le temps.
- `hovering_env` : reward et randomisation de l'environnement LOTF.
- `online` : fenetre de donnees, nombre d'epochs, nombre de hot-swaps, garde-fou anti-divergence.
- `jax` : cache de compilation XLA, genere dans `.jax_cache/` et ignore par git.
- `genesis_bridge` : calibration du pont Genesis et cycle de masses pour le bouton `m`.

Les constantes physiques de la plateforme ne sont pas editees dans le YAML : la source de verite reste `RL-real/cf_params.py` pour la masse nominale, le pas de temps, les bornes d'action, la cible de hover et la structure de l'observation.

## Artefacts et hygiene

`assets/` contient uniquement des exemples presentables : courbes, images et videos. `measurements/` contient les donnees brutes de vol ou de simulation. `models/` contient les checkpoints utiles. Les caches (`.jax_cache/`, `__pycache__/`, `measurements/online_ft/`) et environnements virtuels (`.venv/`) sont regenerables et ignores.
