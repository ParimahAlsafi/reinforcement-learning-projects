"""
Fully Manual Warehouse Robot Environment
--------------------------------------------
This file defines a fully manual environment written without Gymnasium.
It does NOT import gymnasium and does NOT use any ready-made environment.

Scenario:
    The robot starts from S, must first pick up a package P, and then deliver it to G.
    Obstacles X behave like walls: if the robot tries to enter them, it stays in place,
    receives a negative reward, and loses extra energy.

Observation for DQN:
    [row_normalized, col_normalized, has_package, energy_normalized]

Actions:
    0 = LEFT
    1 = DOWN
    2 = RIGHT
    3 = UP
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


Position = Tuple[int, int]

class BoxSpace:
    """Minimal replacement for gymnasium.spaces.Box used only by this project."""
    def __init__(self, low, high, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.dtype = dtype
        self.shape = self.low.shape


class DiscreteSpace:
    """Minimal replacement for gymnasium.spaces.Discrete used only by this project."""
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(self.n))


class WarehouseRobotEnv:
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    ACTIONS = {
        0: (0, -1),   # LEFT
        1: (1, 0),    # DOWN
        2: (0, 1),    # RIGHT
        3: (-1, 0),   # UP
    }

    ACTION_NAMES = {
        0: "LEFT",
        1: "DOWN",
        2: "RIGHT",
        3: "UP",
    }

    ACTION_ARROWS = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑",
    }

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        start_pos: Position = (0, 0),
        package_pos: Position = (2, 2),
        goal_pos: Position = (4, 4),
        obstacles: Optional[List[Position]] = None,
        max_energy: int = 35,
        max_steps: int = 80,
        step_penalty: float = -0.20,
        obstacle_penalty: float = -5.0,
        obstacle_energy_cost: int = 4,
        package_reward: float = 8.0,
        goal_reward: float = 30.0,
        wrong_goal_penalty: float = -2.0,
        energy_empty_penalty: float = -20.0,
        distance_shaping_weight: float = 0.08,
        revisit_penalty: float = -0.30,
        render_mode: Optional[str] = None,
        cell_size: int = 90,
        sleep_time: float = 0.15,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.start_pos = tuple(start_pos)
        self.package_pos = tuple(package_pos)
        self.goal_pos = tuple(goal_pos)
        self.obstacles = set(obstacles or [(1, 1), (1, 3), (3, 1), (3, 3)])

        self.max_energy = int(max_energy)
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.obstacle_penalty = float(obstacle_penalty)
        self.obstacle_energy_cost = int(obstacle_energy_cost)
        self.package_reward = float(package_reward)
        self.goal_reward = float(goal_reward)
        self.wrong_goal_penalty = float(wrong_goal_penalty)
        self.energy_empty_penalty = float(energy_empty_penalty)
        self.distance_shaping_weight = float(distance_shaping_weight)
        self.revisit_penalty = float(revisit_penalty)

        self.render_mode = render_mode
        self.cell_size = int(cell_size)
        self.sleep_time = float(sleep_time)
        self.window = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.assets = {}
        self.path_trail = []

        # DQN receives a 4-dimensional normalized state:
        # row, col, has_package, energy.
        self.observation_space = BoxSpace(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = DiscreteSpace(4)

        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        self.has_package = False
        self.energy = self.max_energy
        self.steps = 0
        self.total_obstacle_hits = 0
        self.total_wall_hits = 0
        self.total_collision_hits = 0
        self.last_action_name = "NONE"
        self.last_reward = 0.0

        self._validate_layout()

    def _validate_layout(self) -> None:
        special = {
            "start_pos": self.start_pos,
            "package_pos": self.package_pos,
            "goal_pos": self.goal_pos,
        }
        for name, pos in special.items():
            if not self._inside(pos):
                raise ValueError(f"{name}={pos} is outside the grid.")
            if pos in self.obstacles:
                raise ValueError(f"{name}={pos} cannot be inside an obstacle.")

    def _inside(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _manhattan_distance(self, pos_a: Position, pos_b: Position) -> int:
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def _current_target(self) -> Position:
        return self.goal_pos if self.has_package else self.package_pos

    def _get_obs(self) -> np.ndarray:
        row = self.agent_pos[0] / max(1, self.rows - 1)
        col = self.agent_pos[1] / max(1, self.cols - 1)
        has_package = 1.0 if self.has_package else 0.0
        energy = self.energy / max(1, self.max_energy)
        return np.array([row, col, has_package, energy], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        self.has_package = False
        self.energy = self.max_energy
        self.steps = 0
        self.total_obstacle_hits = 0
        self.total_wall_hits = 0
        self.total_collision_hits = 0
        self.last_action_name = "NONE"
        self.last_reward = 0.0
        self.path_trail = [self.start_pos]
        obs = self._get_obs()
        info = self._get_info(obstacle_hit=False, success=False)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        action = int(action)
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions are 0, 1, 2, 3.")

        self.steps += 1
        old_pos = tuple(self.agent_pos.tolist())
        old_target = self._current_target()
        old_distance = self._manhattan_distance(old_pos, old_target)

        reward = self.step_penalty
        terminated = False
        truncated = False
        obstacle_hit = False
        wall_hit = False
        collision_hit = False
        success = False

        dr, dc = self.ACTIONS[action]
        proposed = (old_pos[0] + dr, old_pos[1] + dc)

        self.last_action_name = self.ACTION_NAMES[action]

        # Every action consumes one unit of energy.
        self.energy -= 1

        # Boundary check: trying to go outside the grid works like hitting a wall.
        # Wall hits are counted separately from obstacle hits.
        if not self._inside(proposed):
            wall_hit = True
            collision_hit = True
            self.total_wall_hits += 1
            self.total_collision_hits += 1
            reward += self.obstacle_penalty
            self.energy -= self.obstacle_energy_cost
            proposed = old_pos

        # Obstacle check: obstacles are solid blocks.
        # IMPORTANT: the environment does NOT prevent the action before step().
        # If the agent chooses a move toward an obstacle, the collision is registered here.
        elif proposed in self.obstacles:
            obstacle_hit = True
            collision_hit = True
            self.total_obstacle_hits += 1
            self.total_collision_hits += 1
            reward += self.obstacle_penalty
            self.energy -= self.obstacle_energy_cost
            proposed = old_pos

        self.agent_pos = np.array(proposed, dtype=np.int32)
        new_pos = tuple(self.agent_pos.tolist())

        # Extra penalty for cycling/revisiting cells in the same episode.
        # This makes shorter paths better and discourages the robot from walking around too much.
        if new_pos in self.path_trail and new_pos not in {self.package_pos, self.goal_pos}:
            reward += self.revisit_penalty

        if not self.path_trail or self.path_trail[-1] != new_pos:
            self.path_trail.append(new_pos)

        # Small distance shaping. A correct move is still slightly negative overall
        # because step_penalty is larger than distance_shaping_weight.
        # Therefore the agent is encouraged to reach the target with fewer moves.
        new_target = self._current_target()
        new_distance = self._manhattan_distance(new_pos, new_target)
        reward += self.distance_shaping_weight * (old_distance - new_distance)

        # Package pickup.
        if new_pos == self.package_pos and not self.has_package:
            self.has_package = True
            reward += self.package_reward

        # Goal check.
        if new_pos == self.goal_pos:
            if self.has_package:
                reward += self.goal_reward
                terminated = True
                success = True
            else:
                reward += self.wrong_goal_penalty

        # Energy check.
        if self.energy <= 0 and not terminated:
            self.energy = 0
            reward += self.energy_empty_penalty
            terminated = True

        # Max-step truncation.
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        self.last_reward = float(reward)
        obs = self._get_obs()
        info = self._get_info(obstacle_hit=obstacle_hit, wall_hit=wall_hit, collision_hit=collision_hit, success=success)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _get_info(self, obstacle_hit: bool, success: bool, wall_hit: bool = False, collision_hit: bool = False) -> Dict:
        return {
            "position": tuple(self.agent_pos.tolist()),
            "energy": int(self.energy),
            "max_energy": int(self.max_energy),
            "has_package": bool(self.has_package),
            "obstacle_hit": bool(obstacle_hit),      # True only when the robot tries to enter an X obstacle
            "wall_hit": bool(wall_hit),              # True only when the robot tries to leave the grid
            "collision_hit": bool(collision_hit),    # True for obstacle OR wall collisions
            "total_obstacle_hits": int(self.total_obstacle_hits),
            "total_wall_hits": int(self.total_wall_hits),
            "total_collision_hits": int(self.total_collision_hits),
            "success": bool(success),
            "steps": int(self.steps),
            "action": self.last_action_name,
            "last_reward": float(self.last_reward),
        }

    def render(self):
        if self.render_mode == "ansi":
            return self.render_text()
        if self.render_mode == "human":
            self._render_pygame()
        return None

    def render_text(self) -> str:
        lines = []
        pos = tuple(self.agent_pos.tolist())
        for r in range(self.rows):
            row_symbols = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == pos:
                    row_symbols.append("R")
                elif cell in self.obstacles:
                    row_symbols.append("X")
                elif cell == self.start_pos:
                    row_symbols.append("S")
                elif cell == self.package_pos and not self.has_package:
                    row_symbols.append("P")
                elif cell == self.goal_pos:
                    row_symbols.append("G")
                else:
                    row_symbols.append(".")
            lines.append(" ".join(row_symbols))
        lines.append(
            f"Energy={self.energy}/{self.max_energy} | "
            f"Package={self.has_package} | Steps={self.steps} | "
            f"LastAction={self.last_action_name} | LastReward={self.last_reward:.2f}"
        )
        return "\n".join(lines)

    def print_grid(self):
        print(self.render_text())

    def _load_assets(self) -> None:
        """Load visual assets once. If files are missing, fallback drawing is used."""
        if pygame is None or self.assets:
            return
        asset_dir = Path(__file__).resolve().parent / "assets"
        size = int(self.cell_size * 0.76)
        for name in ["robot", "package", "obstacle", "goal"]:
            path = asset_dir / f"{name}.png"
            if path.exists():
                image = pygame.image.load(str(path)).convert_alpha()
                self.assets[name] = pygame.transform.smoothscale(image, (size, size))

    def _draw_vertical_gradient(self, surface, rect, top_color, bottom_color) -> None:
        x, y, w, h = rect
        h = max(1, h)
        for i in range(h):
            t = i / h
            color = tuple(int(top_color[j] * (1 - t) + bottom_color[j] * t) for j in range(3))
            pygame.draw.line(surface, color, (x, y + i), (x + w, y + i))

    def _cell_center(self, row: int, col: int) -> Tuple[int, int]:
        return (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2,
        )

    def _blit_centered_asset(self, name: str, row: int, col: int) -> bool:
        image = self.assets.get(name)
        if image is None:
            return False
        cx, cy = self._cell_center(row, col)
        rect = image.get_rect(center=(cx, cy))
        # Soft object shadow
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(
            shadow,
            (0, 0, 0, 55),
            (rect.width * 0.15, rect.height * 0.72, rect.width * 0.70, rect.height * 0.22),
        )
        self.window.blit(shadow, rect)
        self.window.blit(image, rect)
        return True

    def _render_pygame(self) -> None:
        if pygame is None:
            raise ImportError("pygame is not installed. Run: pip install pygame")

        pygame.init()
        footer_height = 142
        width = self.cols * self.cell_size
        height = self.rows * self.cell_size + footer_height

        if self.window is None:
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Smart Warehouse Robot - Fully Manual Env + DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 22, bold=True)
            self.small_font = pygame.font.SysFont("arial", 16)
            self._load_assets()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Modern warehouse palette
        bg_top = (222, 235, 247)
        bg_bottom = (190, 207, 223)
        cell_a = (245, 248, 250)
        cell_b = (233, 239, 244)
        grid_line = (142, 158, 171)
        start_color = (205, 235, 255)
        goal_color = (207, 255, 224)
        package_color = (255, 243, 206)
        obstacle_color = (75, 82, 91)
        black = (20, 25, 30)
        white = (255, 255, 255)
        blue = (42, 132, 220)
        green = (30, 190, 105)
        red = (235, 76, 76)
        yellow = (255, 188, 39)
        footer_color = (25, 32, 41)

        self._draw_vertical_gradient(self.window, (0, 0, width, height), bg_top, bg_bottom)

        # Board shadow and base
        board_rect = pygame.Rect(8, 8, width - 16, self.rows * self.cell_size - 16)
        shadow = pygame.Surface((board_rect.width + 18, board_rect.height + 18), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 45), shadow.get_rect(), border_radius=18)
        self.window.blit(shadow, (board_rect.x - 1, board_rect.y + 5))
        pygame.draw.rect(self.window, (227, 233, 240), board_rect, border_radius=18)

        # Draw cells
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size + 10
                y = r * self.cell_size + 10
                rect = pygame.Rect(x, y, self.cell_size - 20, self.cell_size - 20)
                cell = (r, c)
                color = cell_a if (r + c) % 2 == 0 else cell_b
                if cell == self.start_pos:
                    color = start_color
                if cell == self.goal_pos:
                    color = goal_color
                if cell == self.package_pos and not self.has_package:
                    color = package_color
                if cell in self.obstacles:
                    color = obstacle_color

                pygame.draw.rect(self.window, color, rect, border_radius=16)
                pygame.draw.rect(self.window, grid_line, rect, width=2, border_radius=16)

                # Subtle floor texture lines
                if cell not in self.obstacles:
                    pygame.draw.line(self.window, (255, 255, 255, 70), (rect.left + 12, rect.top + 14), (rect.right - 12, rect.top + 14), 1)
                    pygame.draw.line(self.window, (212, 221, 229), (rect.left + 12, rect.bottom - 14), (rect.right - 12, rect.bottom - 14), 1)

                # Start label
                if cell == self.start_pos:
                    label = self.small_font.render("START", True, (24, 100, 155))
                    self.window.blit(label, (rect.left + 7, rect.top + 6))

                # Special objects
                if cell in self.obstacles:
                    if not self._blit_centered_asset("obstacle", r, c):
                        pygame.draw.rect(self.window, red, rect.inflate(-20, -20), border_radius=12)
                elif cell == self.package_pos and not self.has_package:
                    if not self._blit_centered_asset("package", r, c):
                        pygame.draw.rect(self.window, yellow, rect.inflate(-20, -20), border_radius=12)
                elif cell == self.goal_pos:
                    if not self._blit_centered_asset("goal", r, c):
                        pygame.draw.circle(self.window, green, rect.center, self.cell_size // 4)

        # Draw learned/visited path trail of current episode
        if len(self.path_trail) >= 2:
            pts = [self._cell_center(r, c) for r, c in self.path_trail]
            pygame.draw.lines(self.window, (40, 116, 255), False, pts, width=6)
            pygame.draw.lines(self.window, (153, 205, 255), False, pts, width=2)
            for idx, (px, py) in enumerate(pts[:-1]):
                pygame.draw.circle(self.window, (30, 105, 220), (px, py), 7)
                pygame.draw.circle(self.window, white, (px, py), 3)

        # Draw robot with direction
        pos = tuple(self.agent_pos.tolist())
        if "robot" in self.assets:
            robot = self.assets["robot"]
            angle_map = {"UP": 0, "RIGHT": -90, "DOWN": 180, "LEFT": 90}
            angle = angle_map.get(self.last_action_name, 0)
            robot = pygame.transform.rotate(robot, angle)
            cx, cy = self._cell_center(pos[0], pos[1])
            rect = robot.get_rect(center=(cx, cy))
            glow = pygame.Surface((rect.width + 24, rect.height + 24), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (76, 172, 255, 70), glow.get_rect())
            self.window.blit(glow, (rect.x - 12, rect.y - 12))
            self.window.blit(robot, rect)
        else:
            center = self._cell_center(pos[0], pos[1])
            pygame.draw.circle(self.window, blue, center, self.cell_size // 3)
            robot_text = self.font.render("R", True, white)
            self.window.blit(robot_text, robot_text.get_rect(center=center))

        # Footer panel
        footer_y = self.rows * self.cell_size
        pygame.draw.rect(self.window, footer_color, (0, footer_y, width, footer_height))
        pygame.draw.line(self.window, (88, 105, 122), (0, footer_y), (width, footer_y), 2)

        title = self.font.render("DQN Warehouse Robot", True, white)
        self.window.blit(title, (14, footer_y + 10))

        status = "DELIVERING" if self.has_package else "SEARCHING PACKAGE"
        status_color = green if self.has_package else yellow
        status_text = self.small_font.render(
            f"Mode: {status}   |   Step: {self.steps}/{self.max_steps}   |   Action: {self.last_action_name}   |   Reward: {self.last_reward:.2f}",
            True,
            (225, 235, 242),
        )
        self.window.blit(status_text, (14, footer_y + 44))

        package_text = self.small_font.render(f"Package picked: {self.has_package}", True, status_color)
        self.window.blit(package_text, (width - 190, footer_y + 14))

        # Energy bar with label
        bar_x, bar_y = 14, footer_y + 78
        bar_w, bar_h = width - 28, 28
        energy_ratio = max(0.0, min(1.0, self.energy / max(1, self.max_energy)))
        current_w = int(bar_w * energy_ratio)
        bar_color = green if energy_ratio > 0.45 else (255, 164, 42) if energy_ratio > 0.25 else red
        pygame.draw.rect(self.window, (9, 15, 22), (bar_x, bar_y, bar_w, bar_h), border_radius=14)
        pygame.draw.rect(self.window, bar_color, (bar_x, bar_y, current_w, bar_h), border_radius=14)
        pygame.draw.rect(self.window, (150, 166, 183), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=14)
        energy_text = self.small_font.render(f"Energy: {self.energy}/{self.max_energy}", True, white)
        self.window.blit(energy_text, (bar_x + 12, bar_y + 5))

        # Legend
        legend = self.small_font.render("Robot = Agent   |   Box = Package   |   Barricade = Obstacle   |   Green pin = Goal", True, (185, 200, 214))
        self.window.blit(legend, (14, footer_y + 114))

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

    def close(self):
        if pygame is not None and self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
        self.font = None
