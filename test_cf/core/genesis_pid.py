import torch
import math

def get_motors_rpm(exec_actions) :
    return (1 + exec_actions) * 14468.429183500699


def get_rpm_from_pwm(pwm):
    #On a les 2 equations pour le crazyflie2.0 (bitcraze):
    # thrust = 0.409e-3 x pwm² + 140.5e-3 x pwm
    # thrust = 0.109e-6 x rpm² - 210.6e-6 x rpm
    #On resout le polynôme

    # Données par bitcraze
    a_rpm = 0.109e-6
    b_rpm = -210.6e-6

    if not isinstance(pwm, torch.Tensor):
        pwm = torch.tensor([pwm], dtype=torch.float32)

    thrust = 0.409e-3 * (pwm**2) + 140.5e-3 * pwm
    delta = (b_rpm**2) - (4 * a_rpm * (-thrust))
    safe_delta = torch.clamp(delta, min=0.0)
    rpm = (-b_rpm + torch.sqrt(safe_delta)) / (2 * a_rpm)
    
    # Si delta était négatif, on force le RPM à 0
    rpm = torch.where(delta < 0, torch.zeros_like(rpm), rpm)
    return rpm


class PIDTorch:
    def __init__(self, kp, ki, kd, num_envs, device, i_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit

        self.i = torch.zeros(num_envs, device=device)
        self.prev_error = torch.zeros(num_envs, device=device)

    def reset(self, idx):
        self.i[idx] = 0.0
        self.prev_error[idx] = 0.0

    def step(self, error, dt):
        self.i += error * dt
        if self.i_limit is not None:
            self.i = torch.clamp(self.i, -self.i_limit, self.i_limit)

        d = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.i + self.kd * d


class CrazyfliePIDTorch:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs

        # ===== PARAMÈTRES CRAZYFLIE =====
        self.max_rpm = 29000.0
 
        # ===== ATTITUDE → RATE =====
        self.pid_roll = PIDTorch(6, 3, 0, num_envs, device, 20)
        self.pid_pitch = PIDTorch(6, 3, 0, num_envs, device, 20)
        self.pid_yaw = PIDTorch(6, 1, 0.35, num_envs, device, 360)
        #dans crazyflie-firmware/src/platform/interface/platform_defaults_bolt.h

        # ===== RATE → TORQUE =====
        self.pid_p = PIDTorch(250, 500, 2.5, num_envs, device, 33.3)
        self.pid_q = PIDTorch(250, 500, 2.5, num_envs, device, 33.3)
        self.pid_r = PIDTorch(120, 16.7, 0, num_envs, device, 166.7)

    def reset_idx(self, idx):
        self.pid_roll.reset(idx)
        self.pid_pitch.reset(idx)
        self.pid_yaw.reset(idx)
        self.pid_p.reset(idx)
        self.pid_q.reset(idx)
        self.pid_r.reset(idx)

    def update(self, euler_deg, gyro, actions, dt, cascade):
        """
        euler_deg: (N,3) [roll, pitch, yaw] deg
        gyro: (N,3) rad/s
        cascade 2: Actions = [roll_deg, pitch_deg, yaw_rate, thrust_rpm]
        cascade 1: Actions = [p_rate_sp, q_rate_sp, r_rate_sp, thrust_rpm]
        cascade 0: Actions = [rpm1, rpm2, rpm3, rpm4] (Contrôle direct moteurs)
        """

        if cascade == 2:
            # --- ANGLES ---
            roll, pitch = euler_deg[:, 0], euler_deg[:, 1]
            roll_sp, pitch_sp  = actions[:, 0],  actions[:, 1]
            # --- ATTITUDE PID ---
            p_sp = self.pid_roll.step(roll_sp - roll, dt)
            q_sp = self.pid_pitch.step(pitch_sp - pitch, dt)
            r_sp = actions[:, 2] # yaw directement en rate
        
        elif cascade == 1 :
            # Entrée : Rate Setpoints (p, q, r) directement dans les actions
            p_sp, q_sp, r_sp = actions[:, 0], actions[:, 1], actions[:, 2]

        if cascade >= 1:
            # --- RATE PID ---
            tau_x = self.pid_p.step(p_sp - gyro[:, 0], dt)
            tau_y = self.pid_q.step(q_sp - gyro[:, 1], dt)
            tau_z = self.pid_r.step(r_sp - gyro[:, 2], dt)


            # --- THRUST ---
            
            PWM = actions[:,3] * 256/100
            rpm = get_rpm_from_pwm(PWM)

    
            thrust = torch.clamp(rpm, 0, self.max_rpm)

            # --- MIXER CRAZYFLIE (X) ---

            F1 = thrust - tau_x/2 - tau_y/2 - tau_z
            F2 = thrust - tau_x/2 + tau_y/2 + tau_z
            F3 = thrust + tau_x/2 + tau_y/2 - tau_z
            F4 = thrust + tau_x/2 - tau_y/2 + tau_z
            # Configuration en X standard
            
            rpm = torch.stack([F1, F2, F3, F4], dim=1)

            rpm = torch.clamp(rpm, 0.0, self.max_rpm)

            return rpm

        if cascade == 0 :
                
                PWM = actions * 256/100
                rpm = get_rpm_from_pwm(PWM)
                rpm = torch.clamp(rpm, 0.0, self.max_rpm)

                return rpm


   


        
   