
"""Calibration de la chaîne actionneur Genesis : mesure la force verticale RÉELLE
F(pwm%) produite par PWM%→(genesis_pid get_rpm_from_pwm)→set_propellers_rpm→physique
Genesis, puis en déduit la map inverse `thrust_N_to_pwm_pct` correcte.

Problème diagnostiqué : la map actuelle suppose hover=82 % PWM, mais le drone
Genesis plane vers ~68-70 % -> la poussée commandée par la politique est
quasi découplée de la portance réelle. On corrige en mesurant la vraie courbe.

Méthode : drone à plat, immobile, en l'air ; on commande un PWM constant
(rate setpoints nuls -> tous moteurs identiques), on intègre quelques sous-pas
et on lit a_z = Δv_z/Δt sur la fenêtre initiale (avant que la vitesse/traînée
ne montent). Force totale F = m·(a_z + g).
"""
import sys
from pathlib import Path
import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS)); sys.path.insert(0, str(THIS / "RL-real"))
import cf_params as P                              # noqa: E402


def measure_force(env, pwm_pct, mass, n_sub=4, z0=1.5):
    """Force verticale totale (N) produite par un PWM% constant, mesurée par a_z.

    Utilise la MASSE RÉELLEMENT simulée par Genesis -> l'actionneur délivre les
    vrais Newtons commandés (actionneur honnête)."""
    dev = env.device
    env.drone.set_pos(torch.tensor([[0.0, 0.0, z0]], device=dev, dtype=torch.float32),
                      zero_velocity=True)
    env.drone.set_quat(env.base_init_quat.unsqueeze(0), zero_velocity=True)
    env.drone.zero_all_dofs_velocity()
    env.pid.reset_idx(torch.arange(env.num_envs, device=dev))

    from genesis.utils.geom import inv_quat, transform_by_quat
    pid_actions = torch.tensor([[0.0, 0.0, 0.0, pwm_pct]], device=dev, dtype=torch.float32)
    dummy_euler = torch.zeros((env.num_envs, 3), device=dev, dtype=torch.float32)

    vz = [float(env.drone.get_vel().detach().cpu().numpy().reshape(-1)[2])]
    for _ in range(n_sub):
        gyro = transform_by_quat(env.drone.get_ang(), inv_quat(env.drone.get_quat()))
        rpm = env.pid.update(dummy_euler, gyro, pid_actions, env.physics_dt, cascade=1)
        env.drone.set_propellers_rpm(rpm)
        env.scene.step()
        vz.append(float(env.drone.get_vel().detach().cpu().numpy().reshape(-1)[2]))
    # a_z par régression linéaire de v_z(t) sur la fenêtre
    t = np.arange(len(vz)) * env.physics_dt
    a_z = np.polyfit(t, vz, 1)[0]
    return mass * (a_z + P.G)


def main():
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")
    from lotf_genesis_env import LotfHoverEnv
    env = LotfHoverEnv(goal=[0, 0, 0.5], show_viewer=False, visualize_target=False)

    # masse RÉELLEMENT simulée par Genesis (cf2x.urdf) -> actionneur honnête
    genesis_mass = float(sum(l.get_mass() for l in env.drone.links))
    print(f"  masse Genesis simulée = {genesis_mass:.4f} kg "
          f"(cf_params.MASS = {P.MASS} kg -> écart de masse réel)")

    pwms = np.arange(40.0, 100.1, 5.0)
    forces = np.array([measure_force(env, float(p), genesis_mass) for p in pwms])

    print(f"  PWM%   F_genesis(N)   (T_hover = {P.T_HOVER:.4f} N)")
    for p, f in zip(pwms, forces):
        print(f"  {p:5.0f}   {f:9.4f}")

    # fit F(pwm) = a*pwm^2 + b*pwm + c  (force HONNÊTE en N, quadratique en commande)
    a, b, c = np.polyfit(pwms, forces, 2)
    print(f"\n  fit F(pwm) = {a:.6e}*pwm^2 + {b:.6e}*pwm + {c:.6e}")

    def pwm_at(F):
        roots = np.roots([a, b, c - F])
        return float(min([r.real for r in roots if 40 <= r.real <= 100], default=np.nan))

    F_cmd_hover = P.T_HOVER                 # ce que la politique commande au hover (0.027*g)
    F_true_hover = genesis_mass * P.G       # ce qu'il FAUT vraiment pour planer (0.042*g)
    print(f"  >>> PWM pour la force commandée hover ({F_cmd_hover:.3f} N) = {pwm_at(F_cmd_hover):.1f}%")
    print(f"  >>> PWM pour le VRAI hover ({F_true_hover:.3f} N) = {pwm_at(F_true_hover):.1f}%")
    print(f"  => actionneur honnête : la politique 0.027 commandera {F_cmd_hover:.3f} N mais il en "
          f"faut {F_true_hover:.3f} N -> le drone SAGGE (résidu de masse à finetuner).")

    np.savez(THIS / "thrust_calib.npz", pwms=pwms, forces=forces, coef=[a, b, c],
             genesis_mass=genesis_mass)
    print(f"  calibration -> thrust_calib.npz")


if __name__ == "__main__":
    main()
