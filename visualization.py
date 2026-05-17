"""
Visualization using Pygame
==========================

Real-time environment rendering with pygame.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from dynamics.vehicle_params import VehicleParams


class Renderer:
    """Pygame-based environment renderer."""

    def __init__(
        self, arena_size: float, params: VehicleParams, window_size: int = 800
    ):
        """
        Args:
            arena_size: Arena boyutu (m)
            params: Araç parametreleri
            window_size: Window size in pixels
        """
        pygame.init()
        self.arena_size = arena_size
        self.params = params
        self.window_size = window_size
        self.pixels_per_meter = window_size / arena_size

        self.display: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None

        self._initialized = False

    def initialize(self):
        """Initialize pygame display and resources."""
        if not self._initialized:
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("RaceCar RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self._initialized = True

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int(x * self.pixels_per_meter)
        py = int(self.window_size - y * self.pixels_per_meter)
        return px, py

    def _draw_rotated_rect(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        width: int,
        height: int,
        angle: float,
        color: Tuple[int, int, int],
    ):
        """Draw a rotated rectangle."""
        angle_deg = -np.rad2deg(angle)
        rect = pygame.Rect(
            center[0] - width // 2, center[1] - height // 2, width, height
        )
        original = pygame.Surface((width, height))
        original.fill(color)
        rotated = pygame.transform.rotate(original, angle_deg)
        rotated_rect = rotated.get_rect(center=center)
        surface.blit(rotated, rotated_rect)

    def render(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        trajectory: List[Tuple[float, float]],
        step_count: int,
        goal_threshold: float,
        obstacles: List[Dict],
    ) -> Optional[np.ndarray]:
        """
        Render the environment scene.

        Args:
            state: Araç durumu [9]
            goal: Hedef pozisyonu [2]
            trajectory: Geçmiş pozisyonlar listesi
            step_count: Mevcut adım sayısı
            goal_threshold: Hedef eşiği

        Returns:
            None (or RGB array if requested)
        """
        self.initialize()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        X, Y, psi, vx, vy, r, delta, omega_delta, omega_W = state

        self.display.fill((240, 240, 240))

        # Draw arena
        self._draw_arena()

        # Draw trajectory
        self._draw_trajectory(trajectory)

        # Draw goal
        self._draw_goal(goal, goal_threshold)

        # Draw vehicle and obstacles
        self._draw_vehicle(X, Y, psi, delta)
        self._draw_obstacles(obstacles)

        # Draw info
        dist_to_goal = np.sqrt((goal[0] - X) ** 2 + (goal[1] - Y) ** 2)
        self._draw_info(X, Y, psi, vx, delta, step_count, dist_to_goal)

        pygame.display.flip()
        self.clock.tick(30)

        return None

    def _draw_arena(self):
        """Draw arena boundaries."""
        x0, y0 = self._world_to_pixel(0, 0)
        x1, y1 = self._world_to_pixel(self.arena_size, self.arena_size)

        pygame.draw.rect(self.display, (200, 200, 200), (x0, y1, x1 - x0, y0 - y1), 2)

        grid_step = 2.0
        for i in np.arange(0, self.arena_size + grid_step, grid_step):
            px_x, py_x = self._world_to_pixel(i, 0)
            px_y, py_y = self._world_to_pixel(i, self.arena_size)
            pygame.draw.line(
                self.display, (220, 220, 220), (px_x, py_x), (px_y, py_y), 1
            )

            px_x, py_x = self._world_to_pixel(0, i)
            px_y, py_y = self._world_to_pixel(self.arena_size, i)
            pygame.draw.line(
                self.display, (220, 220, 220), (px_x, py_x), (px_y, py_y), 1
            )

    def _draw_trajectory(self, trajectory: List[Tuple[float, float]]):
        """Draw trajectory."""
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                p1 = self._world_to_pixel(trajectory[i][0], trajectory[i][1])
                p2 = self._world_to_pixel(trajectory[i + 1][0], trajectory[i + 1][1])
                pygame.draw.line(self.display, (100, 150, 255), p1, p2, 1)

    def _draw_goal(self, goal: np.ndarray, threshold: float):
        """Draw goal."""
        center = self._world_to_pixel(goal[0], goal[1])
        radius = int(threshold * self.pixels_per_meter)
        pygame.draw.circle(self.display, (100, 200, 100), center, radius, 2)
        pygame.draw.circle(self.display, (0, 255, 0), center, 4)

    def _draw_obstacles(self, obstacles: List[Dict]):
        """Draw dynamic circular obstacles."""
        for obstacle in obstacles:
            pos = obstacle["pos"]
            radius = obstacle["radius"]
            vel = obstacle["vel"]

            center = self._world_to_pixel(pos[0], pos[1])
            px_radius = int(radius * self.pixels_per_meter)
            pygame.draw.circle(self.display, (255, 150, 50), center, px_radius, 2)
            pygame.draw.circle(self.display, (255, 165, 0), center, px_radius)

            vel_scale = 20.0
            vel_end = (
                int(center[0] + vel[0] * vel_scale),
                int(center[1] - vel[1] * vel_scale),
            )
            pygame.draw.line(self.display, (200, 100, 0), center, vel_end, 2)

    def _draw_vehicle(self, X: float, Y: float, psi: float, delta: float):
        """Draw vehicle."""
        car_length = self.params.length
        car_width = self.params.width

        center = self._world_to_pixel(X, Y)
        width_px = int(car_width * self.pixels_per_meter)
        length_px = int(car_length * self.pixels_per_meter)

        self._draw_rotated_rect(
            self.display, center, width_px, length_px, psi, (200, 50, 50)
        )

        arrow_length_px = int(0.5 * self.pixels_per_meter)
        arrow_end = (
            int(center[0] + arrow_length_px * np.cos(psi)),
            int(center[1] - arrow_length_px * np.sin(psi)),
        )
        pygame.draw.line(self.display, (100, 0, 0), center, arrow_end, 2)

        self._draw_front_wheels(X, Y, psi, delta)

    def _draw_front_wheels(self, X: float, Y: float, psi: float, delta: float):
        """Draw front wheels with steering angle."""
        wheel_offset_x = self.params.a
        wheel_offset_y = self.params.width / 2 - 0.03
        wheel_length = 0.1

        for side in [1, -1]:
            wx_body = wheel_offset_x
            wy_body = side * wheel_offset_y

            wx = X + wx_body * np.cos(psi) - wy_body * np.sin(psi)
            wy = Y + wx_body * np.sin(psi) + wy_body * np.cos(psi)

            wheel_angle = psi + delta

            p1 = self._world_to_pixel(
                wx - wheel_length / 2 * np.cos(wheel_angle),
                wy - wheel_length / 2 * np.sin(wheel_angle),
            )
            p2 = self._world_to_pixel(
                wx + wheel_length / 2 * np.cos(wheel_angle),
                wy + wheel_length / 2 * np.sin(wheel_angle),
            )
            pygame.draw.line(self.display, (0, 0, 0), p1, p2, 3)

    def _draw_info(
        self,
        X: float,
        Y: float,
        psi: float,
        vx: float,
        delta: float,
        step_count: int,
        dist_to_goal: float,
    ):
        """Draw info text overlay."""
        info_lines = [
            f"Step: {step_count}",
            f"Pos: ({X:.2f}, {Y:.2f}) m",
            f"Heading: {np.rad2deg(psi):.1f}°",
            f"Velocity: {vx:.2f} m/s",
            f"Steering: {np.rad2deg(delta):.1f}°",
            f"Goal distance: {dist_to_goal:.2f} m",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = self.small_font.render(line, True, (0, 0, 0))
            self.display.blit(text_surface, (10, y_offset))
            y_offset += 22

        title_surface = self.font.render("RaceCar RL", True, (50, 50, 50))
        self.display.blit(title_surface, (self.window_size - 180, 10))

    def close(self):
        """Close pygame and clean up resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
