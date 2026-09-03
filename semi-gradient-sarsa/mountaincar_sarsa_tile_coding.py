
# ============================================================
# Reinforcement Learning Assignment 3
# Mountain Car with Semi-Gradient SARSA + Tile Coding
#
# Question 1: AIUT_Environments MountainCar
# Question 2: Gymnasium MountainCar-v0 with GIF rendering
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
# import imageio

try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Imports for Question 1 and Question 2


# Question 1
import AIUT_Environments as environments

# Question 2
try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Gymnasium is not installed.")
    print("Install it using: pip install gymnasium[classic-control]")


# 1. Global Configuration


RUN_QUESTION_1_AIUT = True
RUN_QUESTION_2_GYMNASIUM = True

NUM_RUNS = 10
NUM_EPISODES = 1000

NUM_TILINGS = 8
TILES_PER_DIM = 8
IHT_SIZE = 4096

ALPHA = 0.5
GAMMA = 1.0

# Epsilon decay settings
USE_EPSILON_DECAY = True
EPSILON_START = 0.1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Evaluation settings
EVAL_EPISODES = 30
SELECTION_WINDOW = 100

# Plot settings
PLOT_RESOLUTION = 80
MOVING_AVERAGE_WINDOW = 50

# Output folder
OUTPUT_DIR = "mountain_car_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Truncation handling
# For AIUT, treating truncation as episode end is simpler and close to assignment style.
BOOTSTRAP_ON_TRUNCATION_AIUT = False

# For Gymnasium, this is theoretically more precise because truncated=True means time limit.
BOOTSTRAP_ON_TRUNCATION_GYM = True


# 2. Utility Functions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def epsilon_schedule(
    episode,
    epsilon_start=0.1,
    epsilon_min=0.01,
    decay_rate=0.995
):
    if not USE_EPSILON_DECAY:
        return epsilon_start

    return max(epsilon_min, epsilon_start * (decay_rate ** episode))


def moving_average(data, window_size=50):
    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )

# 3. Tile Coding Implementation


class IHT:

    def __init__(self, size):
        self.size = size
        self.dictionary = {}
        self.overfull_count = 0

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, readonly=False):
        if obj in self.dictionary:
            return self.dictionary[obj]

        if readonly:
            return None

        if self.full():
            if self.overfull_count == 0:
                print("Warning: IHT is full. Hash collisions may occur.")
            self.overfull_count += 1
            return hash(obj) % self.size

        index = self.count()
        self.dictionary[obj] = index
        return index


def tiles(iht, num_tilings, floats, ints=None, readonly=False):
 

    if ints is None:
        ints = []

    qfloats = [math.floor(f * num_tilings) for f in floats]
    active_tiles = []

    for tiling in range(num_tilings):
        coords = [tiling]
        b = tiling
        tiling_x_2 = tiling * 2

        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tiling_x_2

        coords.extend(ints)
        active_tiles.append(iht.get_index(tuple(coords), readonly))

    return active_tiles

# 4. Semi-Gradient SARSA Agent


class SemiGradientSarsaAgent:


    def __init__(
        self,
        num_tilings=8,
        tiles_per_dim=8,
        iht_size=4096,
        alpha=0.5,
        gamma=1.0,
        actions=(-1, 0, 1),
        position_bounds=(-1.2, 0.5),
        velocity_bounds=(-0.07, 0.07)
    ):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.iht_size = iht_size

        self.alpha = alpha / num_tilings
        self.gamma = gamma

        self.actions = list(actions)

        self.position_min, self.position_max = position_bounds
        self.velocity_min, self.velocity_max = velocity_bounds

        self.position_scale = tiles_per_dim / (self.position_max - self.position_min)
        self.velocity_scale = tiles_per_dim / (self.velocity_max - self.velocity_min)

        self.iht = IHT(iht_size)
        self.w = np.zeros(iht_size)

    def get_active_tiles(self, state, action):
        position, velocity = state

        # Shifted scaling makes the representation easier to understand.
        scaled_position = (position - self.position_min) * self.position_scale
        scaled_velocity = (velocity - self.velocity_min) * self.velocity_scale

        action_index = self.actions.index(action)

        active_tiles = tiles(
            self.iht,
            self.num_tilings,
            [scaled_position, scaled_velocity],
            [action_index]
        )

        return active_tiles

    def q_value(self, state, action):
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.w[active_tiles])

    def choose_action(self, state, epsilon):

        if random.random() < epsilon:
            return random.choice(self.actions)

        q_values = np.array([
            self.q_value(state, action)
            for action in self.actions
        ])

        max_q = np.max(q_values)

        best_actions = [
            action for action, q in zip(self.actions, q_values)
            if q == max_q
        ]

        return random.choice(best_actions)

    def choose_greedy_action(self, state):
        return self.choose_action(state, epsilon=0.0)

    def update(self, state, action, td_target):


        active_tiles = self.get_active_tiles(state, action)
        current_q = np.sum(self.w[active_tiles])
        td_error = td_target - current_q

        self.w[active_tiles] += self.alpha * td_error


# 5. Question 1: AIUT Environment


def train_one_aiut_run(
    run_seed,
    num_episodes=1000,
    verbose=True
):
   

    set_seed(run_seed)

    env = environments.MountainCar()

    agent = SemiGradientSarsaAgent(
        num_tilings=NUM_TILINGS,
        tiles_per_dim=TILES_PER_DIM,
        iht_size=IHT_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        actions=(-1, 0, 1),
        position_bounds=(-1.2, 0.5),
        velocity_bounds=(-0.07, 0.07)
    )

    steps_per_episode = []
    epsilon_per_episode = []

    for episode in range(num_episodes):

        epsilon = epsilon_schedule(
            episode=episode,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            decay_rate=EPSILON_DECAY
        )

        epsilon_per_episode.append(epsilon)

        state = env.reset()
        action = agent.choose_action(state, epsilon)

        steps = 0

        while True:
            next_state, reward, terminated, truncated = env.step(action)
            steps += 1

            if terminated:
                td_target = reward
                agent.update(state, action, td_target)
                break

            if truncated:
                if BOOTSTRAP_ON_TRUNCATION_AIUT:
                    next_action = agent.choose_action(next_state, epsilon)
                    td_target = reward + agent.gamma * agent.q_value(next_state, next_action)
                else:
                    td_target = reward

                agent.update(state, action, td_target)
                break

            next_action = agent.choose_action(next_state, epsilon)
            td_target = reward + agent.gamma * agent.q_value(next_state, next_action)

            agent.update(state, action, td_target)

            state = next_state
            action = next_action

        steps_per_episode.append(steps)

        if verbose and (episode + 1) % 100 == 0:
            print(
                f"Run seed {run_seed} | Episode {episode + 1:4d} | "
                f"Steps = {steps:3d} | Epsilon = {epsilon:.4f}"
            )

    return agent, np.array(steps_per_episode), np.array(epsilon_per_episode)


def evaluate_aiut_agent_greedy(
    agent,
    num_episodes=30,
    seed=10000
):

    set_seed(seed)

    env = environments.MountainCar()
    evaluation_steps = []

    for episode in range(num_episodes):

        state = env.reset()
        steps = 0

        while True:
            action = agent.choose_greedy_action(state)
            next_state, reward, terminated, truncated = env.step(action)

            steps += 1
            state = next_state

            if terminated or truncated:
                break

        evaluation_steps.append(steps)

    return np.array(evaluation_steps)


def generate_aiut_greedy_trajectory(
    agent,
    seed=2025
):
    

    set_seed(seed)

    env = environments.MountainCar()

    state = env.reset()

    positions = []
    velocities = []
    actions = []

    while True:
        positions.append(state[0])
        velocities.append(state[1])

        action = agent.choose_greedy_action(state)
        actions.append(action)

        next_state, reward, terminated, truncated = env.step(action)

        state = next_state

        if terminated or truncated:
            positions.append(state[0])
            velocities.append(state[1])
            break

    return np.array(positions), np.array(velocities), np.array(actions)


# 6. Question 2: Gymnasium Environment


def train_one_gym_run(
    run_seed,
    num_episodes=1000,
    verbose=True
):
    
    if not GYMNASIUM_AVAILABLE:
        raise ImportError("Gymnasium is not installed.")

    set_seed(run_seed)

    env = gym.make("MountainCar-v0")

    agent = SemiGradientSarsaAgent(
        num_tilings=NUM_TILINGS,
        tiles_per_dim=TILES_PER_DIM,
        iht_size=IHT_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        actions=(0, 1, 2),
        position_bounds=(-1.2, 0.6),
        velocity_bounds=(-0.07, 0.07)
    )

    steps_per_episode = []
    epsilon_per_episode = []

    for episode in range(num_episodes):

        epsilon = epsilon_schedule(
            episode=episode,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            decay_rate=EPSILON_DECAY
        )

        epsilon_per_episode.append(epsilon)

        state, info = env.reset(seed=run_seed * 100000 + episode)

        action = agent.choose_action(state, epsilon)

        steps = 0

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated:
                td_target = reward
                agent.update(state, action, td_target)
                break

            if truncated:
                if BOOTSTRAP_ON_TRUNCATION_GYM:
                    next_action = agent.choose_action(next_state, epsilon)
                    td_target = reward + agent.gamma * agent.q_value(next_state, next_action)
                else:
                    td_target = reward

                agent.update(state, action, td_target)
                break

            next_action = agent.choose_action(next_state, epsilon)
            td_target = reward + agent.gamma * agent.q_value(next_state, next_action)

            agent.update(state, action, td_target)

            state = next_state
            action = next_action

        steps_per_episode.append(steps)

        if verbose and (episode + 1) % 100 == 0:
            print(
                f"Run seed {run_seed} | Episode {episode + 1:4d} | "
                f"Steps = {steps:3d} | Epsilon = {epsilon:.4f}"
            )

    env.close()

    return agent, np.array(steps_per_episode), np.array(epsilon_per_episode)


def evaluate_gym_agent_greedy(
    agent,
    num_episodes=30,
    base_seed=50000
):


    if not GYMNASIUM_AVAILABLE:
        return None

    env = gym.make("MountainCar-v0")
    evaluation_steps = []

    for episode in range(num_episodes):

        state, info = env.reset(seed=base_seed + episode)
        steps = 0

        while True:
            action = agent.choose_greedy_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            steps += 1
            state = next_state

            if terminated or truncated:
                break

        evaluation_steps.append(steps)

    env.close()

    return np.array(evaluation_steps)


def generate_gym_greedy_trajectory(
    agent,
    seed=2025
):


    if not GYMNASIUM_AVAILABLE:
        return None, None, None

    env = gym.make("MountainCar-v0")

    state, info = env.reset(seed=seed)

    positions = []
    velocities = []
    actions = []

    while True:
        positions.append(state[0])
        velocities.append(state[1])

        action = agent.choose_greedy_action(state)
        actions.append(action)

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

        if terminated or truncated:
            positions.append(state[0])
            velocities.append(state[1])
            break

    env.close()

    return np.array(positions), np.array(velocities), np.array(actions)


def create_gym_greedy_gif(
    agent,
    filename="gymnasium_mountain_car_greedy.gif",
    seed=2025,
    max_steps=200,
    fps=30
):
   

    if not GYMNASIUM_AVAILABLE:
        return None, None

    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    state, info = env.reset(seed=seed)

    frames = []
    steps = 0

    while True:
        frame = env.render()
        frames.append(frame)

        action = agent.choose_greedy_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        steps += 1
        state = next_state

        if terminated or truncated or steps >= max_steps:
            frame = env.render()
            frames.append(frame)
            break

    env.close()

    imageio.mimsave(filename, frames, fps=fps)

    print(f"GIF saved as: {filename}")
    print(f"Number of steps in GIF episode: {steps}")

    if IPYTHON_AVAILABLE:
        display(Image(filename=filename))

    return filename, steps


# 7. Plotting Functions


def plot_learning_curve(
    average_steps,
    std_steps,
    title,
    filename
):
    episodes = np.arange(1, len(average_steps) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        average_steps,
        linewidth=1.7,
        label="Average steps"
    )

    plt.fill_between(
        episodes,
        average_steps - std_steps,
        average_steps + std_steps,
        alpha=0.2,
        label="±1 standard deviation"
    )

    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title(title)
    plt.ylim(70, 210)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_smoothed_curve(
    average_steps,
    title,
    filename,
    window_size=50
):
    smoothed = moving_average(average_steps, window_size)
    episodes = np.arange(window_size, len(average_steps) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        smoothed,
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel(f"Moving Average Steps, window={window_size}")
    plt.title(title)
    plt.ylim(70, 210)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_epsilon_curve(
    average_epsilons,
    title,
    filename
):
    episodes = np.arange(1, len(average_epsilons) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        average_epsilons,
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def cost_to_go(agent, position, velocity):
    state = [position, velocity]

    q_values = [
        agent.q_value(state, action)
        for action in agent.actions
    ]

    return -np.max(q_values)


def plot_cost_to_go(
    agent,
    title,
    filename
):
    positions = np.linspace(agent.position_min, agent.position_max, PLOT_RESOLUTION)
    velocities = np.linspace(agent.velocity_min, agent.velocity_max, PLOT_RESOLUTION)

    P, V = np.meshgrid(positions, velocities)
    Z = np.zeros_like(P)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            Z[i, j] = cost_to_go(agent, P[i, j], V[i, j])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        P,
        V,
        Z,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-go")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_greedy_trajectory(
    positions,
    velocities,
    title,
    filename
):
    plt.figure(figsize=(10, 5))

    plt.plot(positions, label="Position")
    plt.plot(velocities, label="Velocity")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_phase_trajectory(
    positions,
    velocities,
    title,
    filename
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        positions,
        velocities,
        marker=".",
        markersize=3
    )

    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_evaluation_scores(
    evaluation_scores,
    title,
    filename
):
    runs = np.arange(1, len(evaluation_scores) + 1)

    plt.figure(figsize=(8, 5))

    plt.bar(runs, evaluation_scores)

    plt.xlabel("Run")
    plt.ylabel("Average Greedy Evaluation Steps")
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# 8. Generic Multi-Run Experiment Function


def run_multiple_experiments(
    experiment_name,
    train_one_run_function,
    evaluate_function,
    trajectory_function,
    num_runs=10,
    num_episodes=1000,
    eval_episodes=30,
    output_prefix="experiment"
):

    print("\n" + "=" * 80)
    print(experiment_name)
    print("=" * 80)

    all_steps = np.zeros((num_runs, num_episodes))
    all_epsilons = np.zeros((num_runs, num_episodes))

    evaluation_scores = []
    training_scores = []

    best_agent = None
    best_eval_score = float("inf")
    best_run_index = None

    for run in range(num_runs):

        print(f"\n================ Run {run + 1} / {num_runs} ================")

        agent, steps, epsilons = train_one_run_function(
            run_seed=run,
            num_episodes=num_episodes,
            verbose=True
        )

        all_steps[run, :] = steps
        all_epsilons[run, :] = epsilons

        train_score = np.mean(steps[-SELECTION_WINDOW:])
        training_scores.append(train_score)

        eval_steps = evaluate_function(
            agent,
            num_episodes=eval_episodes,
            seed=50000 + run * 1000
        ) if output_prefix.startswith("q1") else evaluate_function(
            agent,
            num_episodes=eval_episodes,
            base_seed=50000 + run * 1000
        )

        eval_score = np.mean(eval_steps)
        evaluation_scores.append(eval_score)

        print(f"Run {run + 1} | Average steps in last {SELECTION_WINDOW} episodes = {train_score:.2f}")
        print(f"Run {run + 1} | Greedy evaluation average over {eval_episodes} episodes = {eval_score:.2f}")

        if eval_score < best_eval_score:
            best_eval_score = eval_score
            best_agent = agent
            best_run_index = run

    average_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)
    average_epsilons = np.mean(all_epsilons, axis=0)

    training_scores = np.array(training_scores)
    evaluation_scores = np.array(evaluation_scores)

    print("\n" + "=" * 80)
    print(f"{experiment_name} Summary")
    print("=" * 80)
    print(f"Average steps in first 10 episodes          : {np.mean(average_steps[:10]):.2f}")
    print(f"Average steps in last 10 episodes           : {np.mean(average_steps[-10:]):.2f}")
    print(f"Average steps in first 100 episodes         : {np.mean(average_steps[:100]):.2f}")
    print(f"Average steps in last 100 episodes          : {np.mean(average_steps[-100:]):.2f}")
    print(f"Mean training score over runs               : {np.mean(training_scores):.2f}")
    print(f"Std training score over runs                : {np.std(training_scores):.2f}")
    print(f"Mean greedy evaluation score over runs      : {np.mean(evaluation_scores):.2f}")
    print(f"Std greedy evaluation score over runs       : {np.std(evaluation_scores):.2f}")
    print(f"Best run based on greedy evaluation         : Run {best_run_index + 1}")
    print(f"Best greedy evaluation score                : {best_eval_score:.2f}")
    print("=" * 80)

    # Plots
    plot_learning_curve(
        average_steps,
        std_steps,
        title=f"{experiment_name}\nAverage Learning Curve over {num_runs} Runs",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_learning_curve.png")
    )

    plot_smoothed_curve(
        average_steps,
        title=f"{experiment_name}\nSmoothed Learning Curve",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_smoothed_learning_curve.png"),
        window_size=MOVING_AVERAGE_WINDOW
    )

    plot_epsilon_curve(
        average_epsilons,
        title=f"{experiment_name}\nEpsilon Schedule",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_epsilon_curve.png")
    )

    plot_evaluation_scores(
        evaluation_scores,
        title=f"{experiment_name}\nGreedy Evaluation Scores",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_greedy_evaluation_scores.png")
    )

    plot_cost_to_go(
        best_agent,
        title=f"{experiment_name}\nCost-to-go Function of Best Model",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_cost_to_go.png")
    )

    positions, velocities, actions = trajectory_function(best_agent, seed=2025)

    plot_greedy_trajectory(
        positions,
        velocities,
        title=f"{experiment_name}\nOne Greedy Trajectory of Best Model",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_greedy_trajectory.png")
    )

    plot_phase_trajectory(
        positions,
        velocities,
        title=f"{experiment_name}\nPhase Trajectory of Best Model",
        filename=os.path.join(OUTPUT_DIR, f"{output_prefix}_phase_trajectory.png")
    )

    print(f"\nGreedy trajectory length of best model: {len(positions) - 1}")

    return {
        "best_agent": best_agent,
        "best_run_index": best_run_index,
        "best_eval_score": best_eval_score,
        "all_steps": all_steps,
        "average_steps": average_steps,
        "std_steps": std_steps,
        "training_scores": training_scores,
        "evaluation_scores": evaluation_scores
    }


# 9. Main Program


def main():
    print("=" * 80)
    print("Mountain Car Assignment")
    print("Semi-Gradient SARSA + Tile Coding")
    print("=" * 80)
    print(f"Run Question 1 AIUT              : {RUN_QUESTION_1_AIUT}")
    print(f"Run Question 2 Gymnasium         : {RUN_QUESTION_2_GYMNASIUM}")
    print(f"Number of runs                   : {NUM_RUNS}")
    print(f"Number of episodes               : {NUM_EPISODES}")
    print(f"Number of tilings                : {NUM_TILINGS}")
    print(f"Tiles per dimension              : {TILES_PER_DIM}")
    print(f"IHT size                         : {IHT_SIZE}")
    print(f"Alpha before division            : {ALPHA}")
    print(f"Effective alpha                  : {ALPHA / NUM_TILINGS}")
    print(f"Gamma                            : {GAMMA}")
    print(f"Use epsilon decay                : {USE_EPSILON_DECAY}")
    print(f"Epsilon start                    : {EPSILON_START}")
    print(f"Epsilon min                      : {EPSILON_MIN}")
    print(f"Epsilon decay                    : {EPSILON_DECAY}")
    print(f"Evaluation episodes              : {EVAL_EPISODES}")
    print(f"Selection window                 : {SELECTION_WINDOW}")
    print(f"Bootstrap on truncation AIUT     : {BOOTSTRAP_ON_TRUNCATION_AIUT}")
    print(f"Bootstrap on truncation Gym      : {BOOTSTRAP_ON_TRUNCATION_GYM}")
    print(f"Output directory                 : {OUTPUT_DIR}")
    print("=" * 80)

    q1_results = None
    q2_results = None

    if RUN_QUESTION_1_AIUT:
        q1_results = run_multiple_experiments(
            experiment_name="Question 1 - AIUT MountainCar",
            train_one_run_function=train_one_aiut_run,
            evaluate_function=evaluate_aiut_agent_greedy,
            trajectory_function=generate_aiut_greedy_trajectory,
            num_runs=NUM_RUNS,
            num_episodes=NUM_EPISODES,
            eval_episodes=EVAL_EPISODES,
            output_prefix="q1_aiut"
        )

    if RUN_QUESTION_2_GYMNASIUM:
        if GYMNASIUM_AVAILABLE:
            q2_results = run_multiple_experiments(
                experiment_name="Question 2 - Gymnasium MountainCar-v0",
                train_one_run_function=train_one_gym_run,
                evaluate_function=evaluate_gym_agent_greedy,
                trajectory_function=generate_gym_greedy_trajectory,
                num_runs=NUM_RUNS,
                num_episodes=NUM_EPISODES,
                eval_episodes=EVAL_EPISODES,
                output_prefix="q2_gymnasium"
            )

            gif_filename = os.path.join(
                OUTPUT_DIR,
                "q2_gymnasium_best_model_greedy.gif"
            )

            create_gym_greedy_gif(
                q2_results["best_agent"],
                filename=gif_filename,
                seed=2025,
                max_steps=200,
                fps=30
            )
        else:
            print("Gymnasium is not available. Skipping Question 2.")

    print("\nAll experiments finished.")
    print(f"All figures and GIF files are saved in: {OUTPUT_DIR}")

    return q1_results, q2_results


if __name__ == "__main__":
    q1_results, q2_results = main()