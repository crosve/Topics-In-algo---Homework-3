"""
Helper Functions
=====================
"""

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to the range [-pi, pi]

    Args:
        angle: Angle (rad)

    Returns:
        Normalized angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def world_to_body_frame(
    dx_world: float, dy_world: float, psi: float
) -> Tuple[float, float]:
    """
    Transform from world frame to body frame

    Args:
        dx_world: X distance (world frame)
        dy_world: Y distance (world frame)
        psi: Vehicle yaw angle

    Returns:
        (dx_body, dy_body): Distances in body frame
    """
    cos_psi = np.cos(-psi)
    sin_psi = np.sin(-psi)

    dx_body = dx_world * cos_psi - dy_world * sin_psi
    dy_body = dx_world * sin_psi + dy_world * cos_psi

    return dx_body, dy_body


def body_to_world_frame(
    dx_body: float, dy_body: float, psi: float
) -> Tuple[float, float]:
    """
    Transform from body frame to world frame

    Args:
        dx_body: X distance (body frame)
        dy_body: Y distance (body frame)
        psi: Vehicle yaw angle

    Returns:
        (dx_world, dy_world): Distances in world frame
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    dx_world = dx_body * cos_psi - dy_body * sin_psi
    dy_world = dx_body * sin_psi + dy_body * cos_psi

    return dx_world, dy_world


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Distance between p1 and p2
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle_to_point(
    from_pos: Tuple[float, float], to_pos: Tuple[float, float], current_heading: float
) -> float:
    """
    Calculate angle to a point (relative to vehicle heading)

    Args:
        from_pos: Starting position (x, y)
        to_pos: Target position (x, y)
        current_heading: Current heading angle (rad)

    Returns:
        Angle to target (rad), in the range [-pi, pi]
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    angle_to_target = np.arctan2(dy, dx)
    relative_angle = normalize_angle(angle_to_target - current_heading)

    return relative_angle


def erpm_to_velocity(
    erpm: float, wheel_radius: float, motor_poles: int, gear_ratio: float
) -> float:
    """Convert ERPM to velocity"""
    return erpm * 2 * np.pi * wheel_radius / (60 * motor_poles * gear_ratio)


def velocity_to_erpm(
    velocity: float, wheel_radius: float, motor_poles: int, gear_ratio: float
) -> float:
    """Convert velocity to ERPM"""
    return velocity * 60 * motor_poles * gear_ratio / (2 * np.pi * wheel_radius)
