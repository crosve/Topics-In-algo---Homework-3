"""Pure pursuit helper for heading reference."""

from typing import List, Tuple
import numpy as np
from utils import normalize_angle


class PurePursuitPlanner:
    """Computes steering/heading references from waypoints."""

    def __init__(self, wheelbase: float):
        self.wheelbase = wheelbase

    def _nearest_index(self, position: np.ndarray, path: np.ndarray) -> int:
        dists = np.linalg.norm(path - position[None, :], axis=1)
        return int(np.argmin(dists))

    def _project_to_segment(
        self,
        position: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        seg = end - start
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            return start.copy(), 0.0, 0.0
        t = float(np.dot(position - start, seg) / seg_len_sq)
        t = float(np.clip(t, 0.0, 1.0))
        projection = start + t * seg
        seg_len = float(np.sqrt(seg_len_sq))
        return projection, t, seg_len

    def get_lookahead_point(
        self,
        position: Tuple[float, float],
        path: List[Tuple[float, float]],
        lookahead_distance: float,
    ) -> np.ndarray:
        path_arr = np.asarray(path, dtype=np.float64)
        if len(path_arr) == 0:
            raise ValueError("Path must include at least one waypoint.")
        if len(path_arr) == 1:
            return path_arr[0]
        # The homework path is often [start, goal]; always target goal to avoid
        # selecting the starting waypoint after the vehicle moves away from it.
        if len(path_arr) == 2:
            return path_arr[1]

        pos = np.asarray(position, dtype=np.float64)

        seg_starts = path_arr[:-1]
        seg_ends = path_arr[1:]
        seg_lengths = np.linalg.norm(seg_ends - seg_starts, axis=1)
        cumulative = np.zeros(len(path_arr), dtype=np.float64)
        cumulative[1:] = np.cumsum(seg_lengths)

        best_seg = 0
        best_t = 0.0
        best_dist = float("inf")
        for i in range(len(seg_starts)):
            proj, t, _ = self._project_to_segment(pos, seg_starts[i], seg_ends[i])
            dist = float(np.linalg.norm(proj - pos))
            if dist < best_dist:
                best_dist = dist
                best_seg = i
                best_t = t

        progress = cumulative[best_seg] + best_t * (
            cumulative[best_seg + 1] - cumulative[best_seg]
        )
        target_progress = progress + max(0.0, float(lookahead_distance))

        if target_progress >= cumulative[-1]:
            return path_arr[-1]

        target_seg = int(np.searchsorted(cumulative, target_progress, side="right") - 1)
        target_seg = int(np.clip(target_seg, 0, len(seg_starts) - 1))
        seg_len = cumulative[target_seg + 1] - cumulative[target_seg]
        if seg_len < 1e-12:
            return path_arr[target_seg + 1]
        ratio = (target_progress - cumulative[target_seg]) / seg_len
        return path_arr[target_seg] + ratio * (
            path_arr[target_seg + 1] - path_arr[target_seg]
        )

    def compute_heading_reference(
        self,
        position: Tuple[float, float],
        heading: float,
        path: List[Tuple[float, float]],
        lookahead_distance: float,
    ) -> float:
        target = self.get_lookahead_point(position, path, lookahead_distance)
        desired_heading = np.arctan2(target[1] - position[1], target[0] - position[0])
        return normalize_angle(desired_heading - heading)

    def heading_error_to_delta(
        self, heading_error: float, speed: float, lookahead_distance: float
    ) -> float:
        speed = max(speed, 0.5)
        ld = max(lookahead_distance, 0.5)

        # U-Turn heuristic: If goal is behind, force max steering
        if np.cos(heading_error) < 0:
            heading_error = np.pi / 2 if heading_error > 0 else -np.pi / 2

        curvature = 2.0 * np.sin(heading_error) / ld
        return np.arctan(self.wheelbase * curvature)
