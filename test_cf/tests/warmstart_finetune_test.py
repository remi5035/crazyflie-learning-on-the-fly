"""Teste si le FINETUNE (depuis base27, résidu de masse) apprend le biais de hover.

Utilise la mécanique EXACTE de config 3 (`finetune_lotf_jax.run_bptt` : warm-start
depuis base27, env 27 g + a_res=coeff·f_d·R_z, BPTT court horizon 3 s, lr 1e-3,
grad-clip 0.5), mais avec le coeff de résidu ANALYTIQUE EXACT (= ce qu'un log parfait
de pur décalage 27->33 g donnerait : coeff=1/0.033-1/0.027) au lieu d'un fit sur log.
C'est donc config 3 avec le MEILLEUR résidu possible. Si le réseau apprend ici le
+0.059 N de hover -> oui, le finetune peut absorber la masse sans connaître m_real.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import jax
import jax.numpy as jnp

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
import finetune_lotf_jax as F                                 # noqa: E402
from lotf_jax_bridge import torch_sd_to_flax, flax_to_torch_sd  # noqa: E402
from eval_in_pretrain_env import eval_policy                   # noqa: E402

M_NOM, M_REAL = 0.027, 0.033
COEFF = 1.0 / M_REAL - 1.0 / M_NOM           # = ce que mass_estimation fitterait sur un log parfait


def main():
    target = list(P.HOVER_GOAL)
    base_path = TEST_CF / "models" / "model_pretrain.pt"
    base = torch_sd_to_flax(torch.load(base_path, map_location="cpu")["model_state_dict"])
    print(f"coeff résidu exact = {COEFF:.3f} (1/kg)  -> m_eff = "
          f"{1.0/(1.0/M_NOM + COEFF)*1e3:.1f} g  (a_res = coeff·f_d·R_z)\n")

    hover = np.array([9.81 * M_NOM, 0.0, 0.0, 0.0])           # env finetune = quad 27 g -> 0.265
    print(f"  {'politique':34s} {'offset z':>9s} {'xy drift':>9s} {'z min':>8s} {'sortie boîte':>13s}")
    # référence : base27 dans l'env 27g+résidu (= AVANT finetune, pour voir le sag de départ)
    for name, rel, mres in [("base27 brute @ 27g+résidu", "models/model_pretrain.pt", M_REAL)]:
        _, z, xy, zmin, left, k = eval_policy(str(TEST_CF / rel), M_NOM, mres, target)
        out = f"pas {k} ({k*P.DT:.1f}s)" if left else "non"
        print(f"  {name:34s} {z:+8.3f}m {xy:8.3f}m {zmin:7.2f}m {out:>13s}")

    for epochs in (30, 90):
        ctx = F.build_bptt_context(target, epochs, res_mode="mass_estimation")
        ft = F.run_bptt(ctx, base, jnp.asarray(COEFF, dtype=jnp.float32))
        out_pt = TEST_CF / "models" / f"model_ft_exact_e{epochs}.pt"
        torch.save({"model_state_dict": flax_to_torch_sd(ft, hover)}, out_pt)
        _, z, xy, zmin, left, k = eval_policy(str(out_pt), M_NOM, M_REAL, target)
        sortie = f"pas {k} ({k*P.DT:.1f}s)" if left else "non"
        # biais de sortie appris (devrait monter vers +0.059 si la masse est absorbée)
        b0 = float(np.asarray(ft["params"]["Dense_2"]["bias"])[0])
        print(f"  {'FINETUNE exact, '+str(epochs)+' epochs':34s} {z:+8.3f}m {xy:8.3f}m {zmin:7.2f}m "
              f"{sortie:>13s}   (actor.4.bias[0]={b0:+.3f})")


if __name__ == "__main__":
    main()
