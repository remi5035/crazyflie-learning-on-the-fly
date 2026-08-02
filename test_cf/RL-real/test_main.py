"""Flight entry point — LOTF-aligned, no runtime tuning knobs."""
import argparse
import logging
import sys
import time
from pathlib import Path
from threading import Event

from pynput import keyboard
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

from RL_policy import RLPolicy
from drone_state import DroneState
from drone_controller import RLDroneController, save_lotf_log
from utils import plot_history, save_history_to_csv, plot_thrust

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = THIS_DIR.parent / "models" / "model_pretrain.pt"
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

deck_attached_event = Event()


def _on_deck(_, value_str):
    if int(value_str):
        deck_attached_event.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help="Policy checkpoint .pt (27→512→512→4).")
    ap.add_argument("--session", type=str, default="run",
                    help="Base name for logs/plots/.npz outputs.")
    ap.add_argument("--online-finetune", action="store_true",
                    help="Enable the in-flight LOTF update: press 'f' mid-flight to "
                         "fit the residual on the last 2 s, BPTT-finetune, and "
                         "hot-swap the policy without landing.")
    args = ap.parse_args()

    cflib.crtp.init_drivers()
    logging.basicConfig(level=logging.ERROR)

    state = DroneState()
    rl_agent = RLPolicy(args.model)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        controller = RLDroneController(scf, state, rl_agent)
        controller.online_finetune = args.online_finetune

        def on_press(key):
            if key == keyboard.Key.enter: controller.is_flying = True
            if key == keyboard.Key.space: controller.fail_safe = True
            if getattr(key, "char", None) == "f": controller.request_finetune()

        def on_release(key):
            if key == keyboard.Key.enter: controller.exit_program = True; return False
            if key == keyboard.Key.space: controller.land_fail_safe = True; return False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=_on_deck)
        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!'); sys.exit(1)

        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        controller.setup_logs()
        controller.run()

        listener.stop()

    # Hot-swap instants converted into each plot's own time reference:
    #   plot_history → relative to RL start (state.start_flight_time)
    #   plot_thrust  → relative to takeoff   (controller.start_time)
    hs_hist = [t - state.start_flight_time for t in controller.hotswap_times
               if state.start_flight_time]
    hs_thrust = [t - controller.start_time for t in controller.hotswap_times
                 if controller.start_time]

    save_lotf_log(f"{args.session}_lotf.npz")
    save_history_to_csv(state.history, controller.history_cmd, f"{args.session}.csv")
    plot_history(state.history, controller.history_cmd, f"{args.session}_traj.png",
                 hotswaps=hs_hist)
    plot_thrust(controller.thrust_history, f"{args.session}_thrust.png",
                hotswaps=hs_thrust)


if __name__ == '__main__':
    main()
