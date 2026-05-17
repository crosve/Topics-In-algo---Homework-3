"""
Vehicle physical parameters.
"""

from typing import Optional

import numpy as np

from config import VehicleConfig


class VehicleParams:
    """Vehicle physical parameters."""

    def __init__(self, config: Optional[VehicleConfig] = None):
        if config is None:
            config = VehicleConfig()

        self.config = config
        self.wheelbase = config.wheelbase
        self.wheel_radius = config.wheel_radius
        self.length = config.length
        self.width = config.width
        self.height = config.height
        self.mass = config.mass
        self.a = config.a
        self.b = config.b
        self.cg_height = config.cg_height
        self.Iz = config.Iz if config.Iz else self._calculate_yaw_inertia()
        self.Jw = config.Jw if config.Jw else self._calculate_wheel_inertia()
        self.Jst = config.Jst
        self.Bst = config.Bst
        self.Kst = config.Kst
        self.BL = config.BL
        self.motor_poles = config.motor_poles
        self.gear_ratio = config.gear_ratio
        self.max_erpm = config.max_erpm
        self.max_motor_torque = config.max_motor_torque
        self.max_steering_torque = config.max_steering_torque
        self.max_steering_angle = config.max_steering_angle
        self.max_velocity_erpm = self._erpm_to_velocity(self.max_erpm)
        if config.max_velocity > 0.0:
            self.max_velocity = min(config.max_velocity, self.max_velocity_erpm)
        else:
            self.max_velocity = self.max_velocity_erpm
        self.tire_Bf = config.tire_Bf
        self.tire_Br = config.tire_Br
        self.tire_Cf = config.tire_Cf
        self.tire_Cr = config.tire_Cr
        self.mu = config.mu
        self.rolling_resistance = config.rolling_resistance
        self.aero_coeff = config.aero_coeff
        self.g = config.gravity

    # Calculates the yaw inertia based on mass and dimensions
    def _calculate_yaw_inertia(self) -> float:
        return self.mass * (self.length**2 + self.width**2) / 12.0

    # Calculates the wheel inertia based on an estimated wheel mass and radius
    def _calculate_wheel_inertia(self) -> float:
        wheel_mass = 0.15
        return 0.5 * wheel_mass * self.wheel_radius**2 * 4.0

    # Converts ERPM to linear velocity
    def _erpm_to_velocity(self, erpm: float) -> float:
        return (
            erpm
            * 2.0
            * np.pi
            * self.wheel_radius
            / (60.0 * self.motor_poles * self.gear_ratio)
        )

    # Converts linear velocity to ERPM
    def velocity_to_erpm(self, velocity: float) -> float:
        return (
            velocity
            * 60.0
            * self.motor_poles
            * self.gear_ratio
            / (2.0 * np.pi * self.wheel_radius)
        )
