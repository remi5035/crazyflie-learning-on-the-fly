"""Plotting + CSV helpers for the LOTF Crazyflie port."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mark_hotswaps(ax, hotswaps):
    """Draw a vertical dashed line at each in-flight policy hot-swap."""
    if not hotswaps:
        return
    for i, t_hs in enumerate(hotswaps):
        ax.axvline(t_hs, color='purple', ls='--', lw=1.4, alpha=0.8,
                   label='hot-swap' if i == 0 else None)


def plot_history(history, history_cmd, filename='traj.png', hotswaps=None):
    if not history['time']:
        print("Nothing to plot."); return

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    def pair(lx, ly):
        n = min(len(lx), len(ly)); return lx[:n], ly[:n]

    t_x, x = pair(history['time'], history['x_pos'])
    t_y, y = pair(history['time'], history['y_pos'])
    axs[0].plot(t_x, x, 'r-', label='X')
    axs[0].plot(t_y, y, 'g-', label='Y')
    axs[0].plot(history_cmd['time'], history_cmd['x_pos'], 'r--', alpha=0.5, label='X cmd')
    axs[0].plot(history_cmd['time'], history_cmd['y_pos'], 'g--', alpha=0.5, label='Y cmd')
    _mark_hotswaps(axs[0], hotswaps)
    axs[0].set_ylabel('XY (m)'); axs[0].legend(); axs[0].grid(True)
    axs[0].set_title("Horizontal position")
    axs[0].set_ylim(-0.2, 0.7)

    t_r, r = pair(history['time'], history['roll'])
    t_p, p = pair(history['time'], history['pitch'])
    axs[1].plot(t_r, r, 'm-', label='Roll (°)')
    axs[1].plot(t_p, p, 'r-', label='Pitch (°)')
    _mark_hotswaps(axs[1], hotswaps)
    axs[1].set_ylabel('Angle'); axs[1].legend(); axs[1].grid(True)
    axs[1].set_title("Attitude")
    axs[1].set_ylim(-10, 10)

    axs[2].plot(history['time'], history['z_pos'], 'b-', label='Z')
    axs[2].plot(history_cmd['time'], history_cmd['z_pos'], 'k--', label='Z cmd')
    _mark_hotswaps(axs[2], hotswaps)
    axs[2].set_ylabel('Altitude (m)'); axs[2].set_xlabel('t (s)')
    axs[2].legend(); axs[2].grid(True); axs[2].set_title("Altitude")
    axs[2].set_ylim(0.15, 0.6)

    plt.suptitle('LOTF Crazyflie flight')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.show()


def plot_thrust(thrust_data, filename='thrust.png', hotswaps=None):
    if not thrust_data['time']:
        print("No thrust data."); return
    plt.figure(figsize=(10, 4))
    plt.plot(thrust_data['time'], thrust_data['thrust_N'], color='orange', label='Thrust (N)')
    _mark_hotswaps(plt.gca(), hotswaps)
    plt.xlabel('t (s)'); plt.ylabel('Thrust (N)')
    plt.grid(True); plt.legend()
    plt.title("Commanded thrust")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def save_history_to_csv(history, history_cmd, filename='flight.csv', events=None):
    """Sauve l'historique de vol en CSV.

    `events` (optionnel) : liste d'évènements ponctuels {type, time[, compute_time_s]}
    dont `time` est exprimé dans le MÊME repère que history['time'] (relatif au début
    du vol RL). Chaque évènement est marqué sur la ligne la plus proche via deux
    colonnes ajoutées : `event` (ex. 'finetune_trigger' / 'hotswap') et
    `compute_time_s` (hot-swap − déclenchement). Permet de retracer les instants de
    déclenchement du finetune, de hot-swap et les temps de calcul depuis le CSV.
    """
    df_real = pd.concat([pd.Series(v, name=k) for k, v in history.items()], axis=1)
    df_cmd = pd.concat([pd.Series(v, name=k) for k, v in history_cmd.items()], axis=1)
    df_cmd = df_cmd.rename(columns={c: f"{c}_cmd" for c in df_cmd.columns if c != 'time'})

    if 'time' in df_real and 'time' in df_cmd:
        df = pd.merge_asof(
            df_real.dropna(subset=['time']).sort_values('time'),
            df_cmd.dropna(subset=['time']).sort_values('time'),
            on='time', direction='nearest',
        )
    else:
        df = pd.concat([df_real, df_cmd], axis=1)

    if events:
        df = df.reset_index(drop=True)
        df['event'] = ''
        df['compute_time_s'] = np.nan
        for ev in events:
            if 'time' not in df:
                break
            idx = (df['time'] - ev['time']).abs().idxmin()   # ligne la plus proche
            df.at[idx, 'event'] = ev['type']
            if ev.get('compute_time_s') is not None:
                df.at[idx, 'compute_time_s'] = ev['compute_time_s']

    df.to_csv(filename, index=False)
    print(f"Flight data → {filename}  ({len(df)} rows)")
