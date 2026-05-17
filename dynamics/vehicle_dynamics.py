"""
Vehicle dynamics.
"""

import numpy as np

from dynamics.tire_model import TireModel
from dynamics.vehicle_params import VehicleParams


class VehicleDynamics:
    """9-state racecar dynamics model."""

    IDX_X = 0
    IDX_Y = 1
    IDX_PSI = 2
    IDX_VX = 3
    IDX_VY = 4
    IDX_R = 5
    IDX_DELTA = 6
    IDX_OMEGA_DELTA = 7
    IDX_OMEGA_W = 8
    STATE_SIZE = 9

    def __init__(self, params: VehicleParams):
        self.params = params
        self.tire_model = TireModel(params)

    def compute_state_derivative(
        self, state: np.ndarray, tw: float, tst: float
    ) -> np.ndarray:
        x, y, psi, vx, vy, r, delta, omega_delta, omega_w = state
        p = self.params
        alpha_f = self.tire_model.calculate_slip_angle_front(vx, vy, r, delta)
        alpha_r = self.tire_model.calculate_slip_angle_rear(vx, vy, r)
        fz_front, fz_rear = self.tire_model.calculate_normal_forces(ax=0.0)
        fyf = self.tire_model.calculate_lateral_force_front(alpha_f, fz_front)
        fyr = self.tire_model.calculate_lateral_force_rear(alpha_r, fz_rear)
        fx_drive = np.clip(tw / p.wheel_radius, -p.mu * fz_rear, p.mu * fz_rear)
        if abs(vx) < 1e-3:
            f_res = 0.0
        else:
            f_res = p.mass * p.g * p.rolling_resistance * np.sign(vx)

        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r
        vx_dot = (
            vy * r
            + fx_drive / p.mass
            - (2.0 * fyf * np.sin(delta)) / p.mass
            - f_res / p.mass
        )
        vy_dot = -vx * r + (2.0 * fyf * np.cos(delta)) / p.mass + (2.0 * fyr) / p.mass
        r_dot = (p.a * 2.0 * fyf * np.cos(delta)) / p.Iz - (p.b * 2.0 * fyr) / p.Iz
        delta_dot = omega_delta
        omega_delta_dot = (
            tst / p.Jst - (p.Bst / p.Jst) * omega_delta - (p.Kst / p.Jst) * delta
        )
        t_resistance = p.BL * omega_w + fx_drive * p.wheel_radius
        omega_w_dot = (tw - t_resistance) / p.Jw

        return np.array(
            [
                x_dot,
                y_dot,
                psi_dot,
                vx_dot,
                vy_dot,
                r_dot,
                delta_dot,
                omega_delta_dot,
                omega_w_dot,
            ]
        )

    def step_rk4(
        self, state: np.ndarray, tw: float, tst: float, dt: float
    ) -> np.ndarray:
        k1 = self.compute_state_derivative(state, tw, tst)
        k2 = self.compute_state_derivative(state + 0.5 * dt * k1, tw, tst)
        k3 = self.compute_state_derivative(state + 0.5 * dt * k2, tw, tst)
        k4 = self.compute_state_derivative(state + dt * k3, tw, tst)
        return self._apply_constraints(
            state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )

    def step_euler(
        self, state: np.ndarray, tw: float, tst: float, dt: float
    ) -> np.ndarray:
        derivative = self.compute_state_derivative(state, tw, tst)
        return self._apply_constraints(state + dt * derivative)

    def step(self, state: np.ndarray, tw: float, tst: float, dt: float) -> np.ndarray:
        # Use simpler and much faster Euler integration for RL performance
        return self.step_euler(state, tw, tst, dt)

    def _apply_constraints(self, state: np.ndarray) -> np.ndarray:
        new_state = state.copy()
        new_state[self.IDX_PSI] = self._normalize_angle(new_state[self.IDX_PSI])
        new_state[self.IDX_DELTA] = np.clip(
            new_state[self.IDX_DELTA],
            -self.params.max_steering_angle,
            self.params.max_steering_angle,
        )
        new_state[self.IDX_VX] = np.clip(
            new_state[self.IDX_VX], -self.params.max_velocity, self.params.max_velocity
        )
        new_state[self.IDX_OMEGA_W] = max(new_state[self.IDX_OMEGA_W], 0.0)
        return new_state

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    @staticmethod
    def create_initial_state(
        x: float = 0.0, y: float = 0.0, psi: float = 0.0
    ) -> np.ndarray:
        return np.array([x, y, psi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
