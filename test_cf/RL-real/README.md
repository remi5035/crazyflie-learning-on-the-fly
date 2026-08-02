# RL-real — Learning on the Fly, Crazyflie port

Port minimal du papier **Pan et al., RA-L / ICRA 2026** pour piloter un
Crazyflie 2.x via cflib. Aligné **strictement** sur la config Gazebo de
`scripts/rl_controller_lotf.py` + `lotf/envs/hovering_state_env.py`.

## Méthode (rappel)

1. **Politique de base** entraînée en sim différentiable par **BPTT**.
2. **MLP de dynamique résiduelle** `f_θ(p, R, v, T, ω) → a_res` ajustée hors-ligne
   sur des données réelles (`y = a_mesurée − a_nominale`).
3. **BPTT court** de la politique contre le sim *augmenté* du résiduel.

Boucle : voler → fit résiduel + BPTT → refly → ...

## Configuration LOTF (identique au sim Gazebo)

| Paramètre | Valeur |
|---|---|
| `dt` | 0.02 s (50 Hz) |
| `delay` / FIFO last-actions | 0.04 s → 3 actions bufferisées |
| Obs (27) | `rel_pos_norm(3) ‖ R_flat(9) ‖ v_norm(3) ‖ last_actions_norm(12)` |
| Action (4, SI) | `[thrust_total_N, ωx, ωy, ωz]` |
| `hovering_action` (biais MLP) | `[m·g, 0, 0, 0]` |
| MLP | `27 → 512 → 512 → 4` (tanh) |
| Goal | relative ; `HOVER_GOAL = [0, 0, 0.5]` modifiable dans `cf_params.py` |

Toutes les constantes sont centralisées dans **`cf_params.py`**.

## Fichiers

| Fichier | Rôle |
|---|---|
| `cf_params.py` | constantes plateforme + env (single source of truth) |
| `RL_policy.py` | `Actor` MLP 512×512 + biais `hovering_action` |
| `drone_state.py` | construit l'obs 27-dim normalisée depuis les logs cflib |
| `drone_controller.py` | boucle de vol 50 Hz, conversion SI → cflib, log raw LOTF |
| `test_main.py` | entry point cflib + clavier |
| `utils.py` | tracés (`plot_history`, `plot_thrust`) + CSV |
| `lotf_sim.py` | sim différentiable torch (SI, 27-dim, residual-aware) |
| `lotf_residual.py` | MLP résiduel 19→3 + builder de dataset |
| `pretrain_lotf.py` | BPTT sur sim nominal → `model_pretrain.pt` |
| `eval_lotf_sim.py` | teste un checkpoint sur `lotf_sim` (erreur pos + verdict PLANE) |
| `finetune_lotf.py` | fit résiduel + BPTT court → `model_ft.pt` |

## Dépendances

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch numpy scipy matplotlib pandas cflib pynput
```

## Workflow

```bash
2. Lancer le nouveau pretrain

  Depuis test_cf, avec le venv racine ../.venv (jax+lotf+torch) :

  cd test_cf
  ../.venv/bin/python core/pretrain_lotf_jax.py --out models/model_pretrain_jax.pt
  (défauts : 200 epochs, 200 envs, lr 2e-3, sharpness 3.0, ~9 s de calcul.)



# 1) Pretrain (~1 min CPU) — défauts stables (horizon 80, lr 5e-4, clip 0.5)
python pretrain_lotf.py --epochs 300 --out model_pretrain.pt
#    vérifier qu'il plane sur le sim d'origine :
python eval_lotf_sim.py --ckpt model_pretrain.pt

# 2) Vol de collecte (~25 s, ENTRÉE pour décoller, ESPACE = fail-safe)
python test_main.py --model model_pretrain.pt --session run1
# → run1_lotf.npz   (données SI pour le résiduel)
# → run1.csv        (positions, angles)
# → run1_traj.png   (tracés trajectoire)
# → run1_thrust.png (tracé thrust en N)

# 3) Fit résiduel + BPTT court
python finetune_lotf.py --base model_pretrain.pt --log run1_lotf.npz \
    --epochs 30 --out model_ft.pt

# 4) Re-vol avec la politique finetunée
python test_main.py --model model_ft.pt --session run2
```

Itérer 2 → 4 (mises à jour incrémentales, cf. Sec. III-B du papier).

## Mapping Newtons → Crazyflie

Affine simple, dans `cf_params.thrust_N_to_pwm_pct` :
```
pwm% = 82 + (T_N - T_hover) * (100-82) / (T_max - T_hover)    clippé [0, 100]
```
Si tu changes de plateforme (masse / max thrust), édite `cf_params.py`.





