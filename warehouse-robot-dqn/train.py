"""
Train / test / present the fully manual Warehouse Robot environment using DQN.

Important:
    - The environment is fully manual and does not use Gymnasium.
    - No ready-made RL environment is used.
    - The agent uses Deep Q-Learning, not SARSA.

Main commands:
    Train:
        python v0_warehouse_robot_train.py --mode train --episodes 1500

    Test learned policy with Pygame:
        python v0_warehouse_robot_train.py --mode test --render

    Presentation mode:
        python v0_warehouse_robot_train.py --mode presentation --episodes 1200 --visible-training-episodes 20
        In presentation mode, the LAST visible-training-episodes are shown in Pygame.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# These small DQN experiments run faster and more consistently on CPUs with one thread.
torch.set_num_threads(1)

from v0_dqn_agent import DQN, ReplayBuffer, hard_update, optimize_model, select_action, soft_update
from v0_warehouse_robot_env import WarehouseRobotEnv


@dataclass
class TrainStats:
    rewards: List[float]
    successes: List[int]
    obstacle_hits: List[int]
    lengths: List[int]
    energy_left: List[int]
    losses: List[float]
    epsilons: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(render: bool = False, sleep_time: float = 0.08) -> WarehouseRobotEnv:
    return WarehouseRobotEnv(
        rows=5,
        cols=5,
        start_pos=(0, 0),
        package_pos=(2, 2),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (1, 3), (3, 1), (3, 3)],
        max_energy=35,
        max_steps=80,
        render_mode="human" if render else None,
        sleep_time=sleep_time,
    )


def epsilon_by_episode(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_fraction: float,
) -> float:
    decay_episodes = max(1, int(total_episodes * epsilon_decay_fraction))
    progress = min(1.0, episode / decay_episodes)
    return epsilon_start + progress * (epsilon_end - epsilon_start)

def valid_actions_for_position(env: WarehouseRobotEnv, row: int, col: int) -> List[int]:
    """Actions that do not leave the grid and do not hit a solid obstacle."""
    valid: List[int] = []
    for action, (dr, dc) in env.ACTIONS.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and (nr, nc) not in env.obstacles:
            valid.append(action)
    return valid or list(env.ACTIONS.keys())


def valid_actions_from_env(env: WarehouseRobotEnv) -> List[int]:
    row, col = tuple(env.agent_pos.tolist())
    return valid_actions_for_position(env, row, col)


def obstacle_actions_from_env(env: WarehouseRobotEnv) -> List[int]:
    """Actions that would make the robot try to enter an obstacle cell.

    These actions are used only during exploration so the training logs include
    real obstacle collisions. During exploitation, actions are still selected
    from feasible moves for stable learning.
    """
    row, col = tuple(env.agent_pos.tolist())
    obstacle_actions: List[int] = []
    for action, (dr, dc) in env.ACTIONS.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and (nr, nc) in env.obstacles:
            obstacle_actions.append(action)
    return obstacle_actions


def select_masked_action(
    state: np.ndarray,
    policy_net: DQN,
    valid_actions: List[int],
    epsilon: float,
    device: torch.device,
) -> int:
    """Optional helper: epsilon-greedy action selection with invalid actions masked out.

    This function is kept for experimentation, but the main training loop does NOT
    use it, because masking prevents the robot from hitting obstacles/walls and
    makes the obstacle-hit plot stay zero.
    """
    if random.random() < epsilon:
        return random.choice(valid_actions)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0).detach().cpu().numpy()
        masked = np.full_like(q_values, -1e9, dtype=np.float32)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))


def train_dqn(
    episodes: int,
    policy_net: DQN,
    target_net: DQN,
    device: torch.device,
    model_path: str,
    render_training: bool = False,
    render_last_n_episodes: int = 0,
    visible_sleep_time: float = 0.08,
    lr: float = 5e-4,
    gamma: float = 0.95,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    train_every: int = 4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay_fraction: float = 0.9,
    target_update_every: int = 50,
    soft_target_update: bool = True,
    tau: float = 0.02,
    verbose: bool = True,
    seed: int = 7,
) -> TrainStats:
    set_seed(seed)
    # In presentation mode we keep ONE continuous DQN training loop.
    # The environment is rendered only during the last N episodes, so replay buffer,
    # optimizer, and epsilon schedule remain continuous from beginning to end.
    env = make_env(render=render_training, sleep_time=visible_sleep_time)
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    stats = TrainStats([], [], [], [], [], [], [])
    global_step = 0

    for ep in range(1, episodes + 1):
        if render_last_n_episodes > 0 and ep == max(1, episodes - render_last_n_episodes + 1):
            print(f"\nNow showing the LAST {render_last_n_episodes} DQN training episodes in Pygame...")

        if render_training or (render_last_n_episodes > 0 and ep > episodes - render_last_n_episodes):
            env.render_mode = "human"
            env.sleep_time = visible_sleep_time
        else:
            env.render_mode = None

        state, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        episode_loss_values = []
        success = 0

        epsilon = epsilon_by_episode(
            episode=ep,
            total_episodes=episodes,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_fraction=epsilon_decay_fraction,
        )

        while not done:
            # Exploration mostly uses feasible moves, but sometimes intentionally tries
            # an obstacle action. This creates real obstacle collisions in the logs
            # without making the whole DQN training unstable.
            valid_actions = valid_actions_from_env(env)
            obstacle_actions = obstacle_actions_from_env(env)
            if random.random() < epsilon:
                if obstacle_actions and random.random() < 0.08:
                    action = random.choice(obstacle_actions)
                else:
                    action = random.choice(valid_actions)
            else:
                action = greedy_action(policy_net, state, device, valid_actions=valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = int(info.get("success", False))

            replay_buffer.push(state, action, reward, next_state, done)
            global_step += 1

            if global_step % max(1, train_every) == 0:
                loss_value = optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )
                if loss_value is not None:
                    episode_loss_values.append(loss_value)

                if soft_target_update:
                    soft_update(target_net, policy_net, tau=tau)

            state = next_state
            episode_reward += reward

        if (not soft_target_update) and ep % target_update_every == 0:
            hard_update(target_net, policy_net)

        stats.rewards.append(float(episode_reward))
        stats.successes.append(success)
        stats.obstacle_hits.append(int(info.get("total_obstacle_hits", 0)))
        stats.lengths.append(int(info.get("steps", 0)))
        stats.energy_left.append(int(info.get("energy", 0)))
        stats.epsilons.append(float(epsilon))
        if episode_loss_values:
            stats.losses.append(float(np.mean(episode_loss_values)))
        else:
            stats.losses.append(0.0)

        if verbose and (ep == 1 or ep % max(1, episodes // 10) == 0):
            window = min(100, len(stats.rewards))
            avg_reward = np.mean(stats.rewards[-window:])
            success_rate = np.mean(stats.successes[-window:])
            avg_energy = np.mean(stats.energy_left[-window:])
            avg_hits = np.mean(stats.obstacle_hits[-window:])
            print(
                f"Episode {ep:4d}/{episodes} | "
                f"avg_reward={avg_reward:7.2f} | "
                f"success={success_rate:4.2f} | "
                f"avg_energy_left={avg_energy:5.2f} | "
                f"avg_obstacle_hits={avg_hits:5.2f} | "
                f"epsilon={epsilon:5.3f}"
            )

    env.close()
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(policy_net.state_dict(), model_path)
    return stats


def load_model_if_exists(policy_net: DQN, model_path: str, device: torch.device) -> bool:
    if not os.path.exists(model_path):
        return False
    state_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    return True


def greedy_action(policy_net: DQN, state: np.ndarray, device: torch.device, valid_actions: List[int] | None = None) -> int:
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0).detach().cpu().numpy()
        if valid_actions is not None:
            masked = np.full_like(q_values, -1e9, dtype=np.float32)
            masked[valid_actions] = q_values[valid_actions]
            return int(np.argmax(masked))
        return int(np.argmax(q_values))


def state_from_components(env: WarehouseRobotEnv, row: int, col: int, has_package: bool, energy: int | None = None) -> np.ndarray:
    if energy is None:
        energy = env.max_energy
    return np.array(
        [
            row / max(1, env.rows - 1),
            col / max(1, env.cols - 1),
            1.0 if has_package else 0.0,
            energy / max(1, env.max_energy),
        ],
        dtype=np.float32,
    )


def print_policy_matrix(policy_net: DQN, device: torch.device, has_package: bool = False, title: str = "Policy Matrix") -> None:
    env = make_env(render=False)
    print("\n" + "=" * 70)
    print(title)
    print("State assumption:", "agent HAS package" if has_package else "agent does NOT have package yet")
    print("Legend: S=start, P=package, G=goal, X=obstacle, arrows=best DQN action")
    print("=" * 70)

    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.obstacles:
                cells.append("  X  ")
                continue

            if cell == env.goal_pos and has_package:
                cells.append("  G  ")
                continue

            state = state_from_components(env, r, c, has_package=has_package, energy=env.max_energy)
            valid_actions = valid_actions_for_position(env, r, c)
            action = greedy_action(policy_net, state, device, valid_actions=valid_actions)
            arrow = env.ACTION_ARROWS[action]

            if cell == env.start_pos:
                cells.append(f" S{arrow}  ")
            elif cell == env.package_pos:
                label = "P" if not has_package else "P*"
                cells.append(f" {label}{arrow} ")
            elif cell == env.goal_pos:
                cells.append(f" G{arrow}  ")
            else:
                cells.append(f"  {arrow}  ")
        print("".join(cells))
    env.close()
    print("=" * 70 + "\n")


def run_greedy_episode(
    policy_net: DQN,
    device: torch.device,
    render: bool = True,
    sleep_time: float = 0.25,
    max_steps: int = 80,
) -> Dict:
    env = make_env(render=render, sleep_time=sleep_time)
    state, info = env.reset(seed=123)
    total_reward = 0.0
    path = [tuple(env.agent_pos.tolist())]

    print("\nOptimal / Greedy Path using learned DQN policy")
    print("-" * 95)
    print("step | state -> action -> next_state | reward | energy | package | obstacle_hit | success")
    print("-" * 95)

    for step in range(1, max_steps + 1):
        valid_actions = valid_actions_from_env(env)
        action = greedy_action(policy_net, state, device, valid_actions=valid_actions)
        old_pos = tuple(env.agent_pos.tolist())
        next_state, reward, terminated, truncated, info = env.step(action)
        new_pos = tuple(env.agent_pos.tolist())
        total_reward += reward
        path.append(new_pos)

        print(
            f"{step:4d} | {old_pos} -> {env.ACTION_ARROWS[action]} {env.ACTION_NAMES[action]:5s} -> {new_pos} | "
            f"{reward:7.2f} | {info['energy']:6d} | {str(info['has_package']):7s} | "
            f"{str(info['obstacle_hit']):12s} | {str(info['success'])}"
        )

        state = next_state
        if terminated or truncated:
            break

    print("-" * 95)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Reached goal with package: {bool(info.get('success', False))}")
    print(f"Obstacle hits: {info.get('total_obstacle_hits', 0)}")
    print(f"Energy left: {info.get('energy', 0)}/{info.get('max_energy', env.max_energy)}")
    print(f"Number of moves: {info.get('steps', 0)}")
    print("Path:", path)
    env.close()

    return {
        "total_reward": total_reward,
        "success": bool(info.get("success", False)),
        "obstacle_hits": int(info.get("total_obstacle_hits", 0)),
        "energy_left": int(info.get("energy", 0)),
        "steps": int(info.get("steps", 0)),
        "path": path,
    }


def plot_stats(stats: TrainStats, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def moving_average(values: List[float], window: int = 50) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    plots = [
        ("episode_reward.png", "Episode Reward", stats.rewards, "Reward"),
        ("success_rate.png", "Success Rate Moving Average", moving_average(stats.successes), "Success Rate"),
        ("obstacle_hits.png", "Obstacle Hits per Episode", stats.obstacle_hits, "Obstacle Hits"),
        ("energy_left.png", "Energy Left per Episode", stats.energy_left, "Energy Left"),
        ("episode_length.png", "Episode Length", stats.lengths, "Steps"),
        ("epsilon_decay.png", "Epsilon Decay", stats.epsilons, "Epsilon"),
        ("loss.png", "DQN Loss", stats.losses, "Loss"),
    ]

    for filename, title, values, ylabel in plots:
        plt.figure(figsize=(8, 4.5))
        plt.plot(values)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()


def presentation_mode(args, policy_net: DQN, target_net: DQN, device: torch.device) -> None:
    print("\nPresentation Mode")
    print("Step 1) First, the DQN policy matrix is printed before training.")
    print("Step 2) Training starts with high exploration and gradually shifts to exploitation.")
    print("Step 3) Only the LAST visible-training episodes are shown in the beautified Pygame environment.")
    print("Step 4) Finally, the learned greedy/optimal path is executed in Pygame.\n")

    print_policy_matrix(policy_net, device, has_package=False, title="Initial Policy Matrix BEFORE Training - Before Package")
    print_policy_matrix(policy_net, device, has_package=True, title="Initial Policy Matrix BEFORE Training - After Package")

    visible_episodes = min(args.visible_training_episodes, args.episodes)
    silent_episodes = max(0, args.episodes - visible_episodes)
    if silent_episodes > 0:
        print(f"\nTraining silently for the first {silent_episodes} episodes...")

    stats = train_dqn(
        episodes=args.episodes,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
        model_path=args.model_path,
        render_training=False,
        render_last_n_episodes=visible_episodes,
        visible_sleep_time=args.sleep_time,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        train_every=args.train_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_fraction=args.epsilon_decay_fraction,
        verbose=True,
        seed=args.seed,
    )
    plot_stats(stats, args.output_dir)

    print_policy_matrix(policy_net, device, has_package=False, title="Learned Policy Matrix AFTER Training - Before Package")
    print_policy_matrix(policy_net, device, has_package=True, title="Learned Policy Matrix AFTER Training - After Package")

    print("\nRunning the final learned greedy path in the beautified Pygame environment...")
    run_greedy_episode(policy_net, device, render=True, sleep_time=args.sleep_time)

def random_mode(episodes: int = 3, sleep_time: float = 0.20) -> None:
    env = make_env(render=True, sleep_time=sleep_time)
    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"Episode={ep}, Action={info['action']}, reward={reward:.2f}, "
                f"energy={info['energy']}, obstacle_hit={info['obstacle_hit']}, package={info['has_package']}"
            )
            done = terminated or truncated
        print(f"Random episode {ep}: total_reward={total_reward:.2f}, success={info['success']}")
    env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fully Manual Warehouse Robot using Deep Q-Learning")
    parser.add_argument("--mode", choices=["train", "test", "presentation", "random", "matrix"], default="presentation")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--visible-training-episodes", type=int, default=20, help="Number of LAST training episodes shown in Pygame during presentation mode")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model-path", type=str, default="outputs/dqn_warehouse_robot.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.55)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--sleep-time", type=float, default=0.12)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dummy_env = make_env(render=False)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    policy_net = DQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=64).to(device)
    target_net = DQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=64).to(device)
    hard_update(target_net, policy_net)

    if args.mode in ["test", "matrix"]:
        loaded = load_model_if_exists(policy_net, args.model_path, device)
        if not loaded:
            print(f"Model file not found: {args.model_path}")
            print("Please train first:")
            print("python v0_warehouse_robot_train.py --mode train --episodes 1200")
            return
        hard_update(target_net, policy_net)

    if args.mode == "train":
        stats = train_dqn(
            episodes=args.episodes,
            policy_net=policy_net,
            target_net=target_net,
            device=device,
            model_path=args.model_path,
            render_training=args.render,
            visible_sleep_time=args.sleep_time,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            train_every=args.train_every,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            verbose=True,
            seed=args.seed,
        )
        plot_stats(stats, args.output_dir)
        print(f"\nModel saved to: {args.model_path}")
        print(f"Training plots saved to: {args.output_dir}")
        print_policy_matrix(policy_net, device, has_package=False, title="Learned Policy Matrix - Before Package")
        print_policy_matrix(policy_net, device, has_package=True, title="Learned Policy Matrix - After Package")

    elif args.mode == "test":
        print_policy_matrix(policy_net, device, has_package=False, title="Loaded Policy Matrix - Before Package")
        print_policy_matrix(policy_net, device, has_package=True, title="Loaded Policy Matrix - After Package")
        run_greedy_episode(policy_net, device, render=args.render, sleep_time=args.sleep_time)

    elif args.mode == "presentation":
        presentation_mode(args, policy_net, target_net, device)

    elif args.mode == "random":
        random_mode(episodes=args.visible_training_episodes, sleep_time=args.sleep_time)

    elif args.mode == "matrix":
        print_policy_matrix(policy_net, device, has_package=False, title="Policy Matrix - Before Package")
        print_policy_matrix(policy_net, device, has_package=True, title="Policy Matrix - After Package")


if __name__ == "__main__":
    main()
