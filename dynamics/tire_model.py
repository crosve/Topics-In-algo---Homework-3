"""
Tire model.
"""

from typing import Tuple

import numpy as np

from dynamics.vehicle_params import VehicleParams


class TireModel:
    """Simplified pacejka-style model."""

    def __init__(self, params: VehicleParams):
        self.params = params
        self.min_velocity = 0.1
        self.max_slip_angle = np.pi / 4.0

    def calculate_slip_angle_front(self, vx: float, vy: float, r: float, delta: float) -> float:
        vx_safe = max(abs(vx), self.min_velocity)
        alpha_f = delta - np.arctan2(vy + self.params.a * r, vx_safe)
        return np.clip(alpha_f, -self.max_slip_angle, self.max_slip_angle)

    def calculate_slip_angle_rear(self, vx: float, vy: float, r: float) -> float:
        vx_safe = max(abs(vx), self.min_velocity)
        alpha_r = -np.arctan2(vy - self.params.b * r, vx_safe)
        return np.clip(alpha_r, -self.max_slip_angle, self.max_slip_angle)

    def calculate_lateral_force_front(self, alpha_f: float, fz_front: float) -> float:
        b, c = self.params.tire_Bf, self.params.tire_Cf
        d = self.params.mu * fz_front
        return d * np.sin(c * np.arctan(b * alpha_f))

    def calculate_lateral_force_rear(self, alpha_r: float, fz_rear: float) -> float:
        b, c = self.params.tire_Br, self.params.tire_Cr
        d = self.params.mu * fz_rear
        return d * np.sin(c * np.arctan(b * alpha_r))

    def calculate_normal_forces(self, ax: float = 0.0) -> Tuple[float, float]:
        m, g = self.params.mass, self.params.g
        a, b, h = self.params.a, self.params.b, self.params.cg_height
        wheelbase = a + b
        fz_front = m * g * b / wheelbase - m * ax * h / wheelbase
        fz_rear = m * g * a / wheelbase + m * ax * h / wheelbase
        return max(fz_front, 0.1), max(fz_rear, 0.1)
