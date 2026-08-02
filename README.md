# Learning-on-the-fly Crazyflie branch

Cette branche est reduite au banc d'essai Crazyflie situe dans `test_cf/`.

## Ce que fait ce repo

Ce repo valide la boucle LOTF sur Crazyflie avec un cas controle : un changement
de masse. Le fine-tuning par BPTT adapte la politique de vol en utilisant un
modele residuel analytique de changement de masse, c'est-a-dire un terme
d'acceleration proportionnel a la poussee qui reproduit l'effet d'une masse
effective differente de la masse nominale.

La masse elle-meme n'est pas apprise comme parametre de la politique et n'est
pas integree dans l'observation de la policy. Elle est seulement convertie en
residu de dynamique fourni au simulateur differentiable utilise pendant le BPTT.

La meme infrastructure peut aussi etre utilisee avec un residu d'acceleration
classique, par exemple un MLP residuel ajoute a la dynamique nominale. Ce cas est
plus general, mais il est plus difficile a tester proprement : il faut identifier
un residu fiable depuis les logs, verifier sa generalisation, puis seulement
attribuer les gains ou regressions au BPTT. Le cas masse sert donc de validation
controlee de la mecanique BPTT avant de passer a un residu appris plus large.

## Contenu conserve

```text
.
├── test_cf/          # pipeline Crazyflie : Genesis, vol reel, fine-tuning, resultats
│   └── core/genesis_pid.py  # PID Genesis utilise par lotf_genesis_env.py
├── lotf/             # moteur JAX/LOTF utilise par le pretrain et le fine-tuning
├── modele_drones/    # URDF/meshes Crazyflie pour Genesis
├── pyproject.toml    # environnement JAX/LOTF racine
├── uv.lock
└── LICENSE
```

La documentation operationnelle est dans `test_cf/README.md`.

## Demarrage rapide

```bash
uv sync
cd test_cf
../.venv/bin/python core/pretrain_lotf_jax.py --preset base27 --out models/model_pretrain.pt
```

Pour les commandes completes : pre-entrainement, vol Genesis, fine-tuning Genesis, vol reel et fine-tuning reel, voir `test_cf/README.md`.

## Dossiers retires

Les exemples papier generiques, assets du README original, scripts ROS/Gazebo, soccer, checkpoints globaux et environnements virtuels locaux ont ete retires de cette branche. Les environnements `.venv/` et `genesis_venv/` sont regenerables et ne doivent pas etre versionnes.
