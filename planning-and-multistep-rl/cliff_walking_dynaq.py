import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(precision=3, suppress=True)

# 1) Cliff Walking Environment - Gymnasium Style

class CliffWalkingEnv:

    def __init__(self, rows=5, cols=20, max_steps=1000):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        self.n_states = rows * cols
        self.n_actions = 4

        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)

        self.cliff = set()
        for c in range(1, cols - 1):
            self.cliff.add((rows - 1, c))

        self.actions = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1)    # Left
        }

        self.current_state = self.start
        self.steps = 0

    def state_to_id(self, state):
        row, col = state
        return row * self.cols + col

    def id_to_state(self, state_id):
        row = state_id // self.cols
        col = state_id % self.cols
        return row, col

    def reset(self):

        self.current_state = self.start
        self.steps = 0

        state_id = self.state_to_id(self.current_state)
        info = {}

        return state_id, info

    def step(self, action):


        if action not in self.actions:
            raise ValueError("Invalid action.")

        self.steps += 1

        row, col = self.current_state
        d_row, d_col = self.actions[action]

        new_row = row + d_row
        new_col = col + d_col

        new_row = max(0, min(self.rows - 1, new_row))
        new_col = max(0, min(self.cols - 1, new_col))

        next_state = (new_row, new_col)

        reward = -1.0
        terminated = False
        truncated = False

        if next_state == self.goal:
            reward = -1.0
            terminated = True

        elif next_state in self.cliff:
            reward = -100.0
            next_state = self.start
            terminated = False

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        self.current_state = next_state
        next_state_id = self.state_to_id(next_state)

        info = {}

        return next_state_id, reward, terminated, truncated, info

# 2) Epsilon-Greedy Action Selection

def epsilon_greedy_action(Q, state, epsilon, n_actions, rng):

    if rng.random() < epsilon:
        return rng.integers(n_actions)

    q_values = Q[state]
    max_q = np.max(q_values)

    best_actions = np.flatnonzero(q_values == max_q)

    return rng.choice(best_actions)

# 3) Q-Learning Update

def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):

    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state])

    Q[state, action] += alpha * (target - Q[state, action])

# 4) Dyna-Q Algorithm
def dyna_q(
    env,
    planning_steps,
    episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):


    rng = np.random.default_rng(seed)

    Q = np.zeros((env.n_states, env.n_actions))

    model = {}
    model_keys = []

    episode_returns = []
    episode_lengths = []

    for episode in range(episodes):

        state, info = env.reset()

        total_reward = 0.0
        steps = 0

        while True:

            action = epsilon_greedy_action(
                Q=Q,
                state=state,
                epsilon=epsilon,
                n_actions=env.n_actions,
                rng=rng
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            q_learning_update(
                Q=Q,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
                alpha=alpha,
                gamma=gamma
            )
            if (state, action) not in model:
                model_keys.append((state, action))

            model[(state, action)] = (next_state, reward, terminated)

            for _ in range(planning_steps):

                random_index = rng.integers(len(model_keys))
                simulated_state, simulated_action = model_keys[random_index]

                simulated_next_state, simulated_reward, simulated_done = model[
                    (simulated_state, simulated_action)
                ]

                q_learning_update(
                    Q=Q,
                    state=simulated_state,
                    action=simulated_action,
                    reward=simulated_reward,
                    next_state=simulated_next_state,
                    done=simulated_done,
                    alpha=alpha,
                    gamma=gamma
                )

            total_reward += reward
            steps += 1

            state = next_state

            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    return Q, episode_returns, episode_lengths

# 5) Moving Average


def moving_average(data, window=20):
    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window) / window, mode="valid")

# 6) Print Final Policy

def print_policy(env, Q):

    arrows = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←"
    }

    for r in range(env.rows):

        row_symbols = []

        for c in range(env.cols):

            state_tuple = (r, c)

            if state_tuple == env.start:
                row_symbols.append("S")

            elif state_tuple == env.goal:
                row_symbols.append("G")

            elif state_tuple in env.cliff:
                row_symbols.append("X")

            else:
                state_id = env.state_to_id(state_tuple)
                best_action = np.argmax(Q[state_id])
                row_symbols.append(arrows[best_action])

        print(" ".join(f"{symbol:>2}" for symbol in row_symbols))

# 7) Print State Values

def print_state_values(env, Q):

    V = np.max(Q, axis=1)

    for r in range(env.rows):

        row_values = []

        for c in range(env.cols):

            state_tuple = (r, c)
            state_id = env.state_to_id(state_tuple)

            if state_tuple == env.goal:
                row_values.append("   G   ")

            elif state_tuple in env.cliff:
                row_values.append("   X   ")

            else:
                row_values.append(f"{V[state_id]:7.3f}")

        print(" ".join(row_values))

# 8) Main Experiment

def main():

    rows = 5
    cols = 20
    max_steps = 1000

    planning_values = [0, 5, 10]

    episodes = 500
    num_runs = 10

    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1

    window = 20

    print("=" * 70)
    print(f"Cliff Walking Environment: {rows}x{cols}")
    print(f"Algorithm: Dyna-Q")
    print(f"Planning steps: {planning_values}")
    print(f"Episodes: {episodes}")
    print(f"Runs per n: {num_runs}")
    print("=" * 70)

    results = {}

    for n in planning_values:

        print("\n" + "=" * 70)
        print(f"Running Dyna-Q with planning steps n = {n}")
        print("=" * 70)

        all_returns = []
        all_lengths = []

        final_Q = None

        for run in range(num_runs):

            env = CliffWalkingEnv(rows=rows, cols=cols, max_steps=max_steps)

            Q, returns, lengths = dyna_q(
                env=env,
                planning_steps=n,
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                seed=run
            )

            all_returns.append(returns)
            all_lengths.append(lengths)

            final_Q = Q

        mean_returns = np.mean(all_returns, axis=0)
        mean_lengths = np.mean(all_lengths, axis=0)

        results[n] = {
            "Q": final_Q,
            "mean_returns": mean_returns,
            "mean_lengths": mean_lengths
        }

        avg_return_last_100 = np.mean(mean_returns[-100:])
        avg_steps_last_100 = np.mean(mean_lengths[-100:])

        print(f"\nn = {n}")
        print(f"Average return in last 100 episodes: {avg_return_last_100:.3f}")
        print(f"Average steps in last 100 episodes: {avg_steps_last_100:.3f}")

        print("\nFinal learned policy:")
        print_policy(env, final_Q)

        print("\nState values V(s) = max_a Q(s,a):")
        print_state_values(env, final_Q)

    # 9) Plot Average Number of Steps

    plt.figure(figsize=(11, 5))

    for n in planning_values:

        lengths = results[n]["mean_lengths"]
        smoothed_lengths = moving_average(lengths, window=window)

        x = np.arange(len(smoothed_lengths)) + window

        plt.plot(x, smoothed_lengths, label=f"Dyna-Q, n = {n}")

    plt.xlabel("Episode")
    plt.ylabel("Average Number of Steps")
    plt.title("Average Number of Steps for Dyna-Q on Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

    # 10) Plot Average Return

    plt.figure(figsize=(11, 5))

    for n in planning_values:

        returns = results[n]["mean_returns"]
        smoothed_returns = moving_average(returns, window=window)

        x = np.arange(len(smoothed_returns)) + window

        plt.plot(x, smoothed_returns, label=f"Dyna-Q, n = {n}")

    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Average Return for Dyna-Q on Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

# 11) Run Program

if __name__ == "__main__":
    main()