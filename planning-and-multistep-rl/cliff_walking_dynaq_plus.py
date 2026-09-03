import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(precision=3, suppress=True)

# 1) Changing Cliff Walking Environment - Gymnasium Style

class ChangingCliffWalkingEnv:

    def __init__(self, rows=5, cols=20, max_steps=1000, change_episode=250):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.change_episode = change_episode

        self.n_states = rows * cols
        self.n_actions = 4

        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)

        self.actions = {
            0: (-1, 0),   
            1: (0, 1),    
            2: (1, 0),    
            3: (0, -1)    
        }

        self.cliff = set()
        for c in range(1, cols - 1):
            self.cliff.add((rows - 1, c))


        self.initial_walls = {
            (rows - 2, 10),
            (rows - 3, 10)
        }

        self.walls = set(self.initial_walls)

        self.current_episode = 0
        self.current_state = self.start
        self.steps = 0

    def state_to_id(self, state):
        row, col = state
        return row * self.cols + col

    def id_to_state(self, state_id):
        row = state_id // self.cols
        col = state_id % self.cols
        return row, col

    def set_episode(self, episode):
        
        self.current_episode = episode

        if episode < self.change_episode:
            self.walls = set(self.initial_walls)
        else:
            self.walls = set()

    def reset(self):
    
        self.current_state = self.start
        self.steps = 0

        state_id = self.state_to_id(self.current_state)

        info = {
            "episode": self.current_episode,
            "changed": self.current_episode >= self.change_episode
        }

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

        if next_state in self.walls:
            next_state = self.current_state
            reward = -1.0

        elif next_state in self.cliff:
            reward = -100.0
            next_state = self.start
            terminated = False

        elif next_state == self.goal:
            reward = -1.0
            terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        self.current_state = next_state
        next_state_id = self.state_to_id(next_state)

        info = {
            "changed": self.current_episode >= self.change_episode
        }

        return next_state_id, reward, terminated, truncated, info


# 2) Epsilon-Greedy Action Selection

def epsilon_greedy_action(Q, state, epsilon, n_actions, rng):
    """
    Select action using epsilon-greedy policy.
    """

    if rng.random() < epsilon:
        return rng.integers(n_actions)

    q_values = Q[state]
    max_q = np.max(q_values)

    best_actions = np.flatnonzero(q_values == max_q)

    return rng.choice(best_actions)


# 3) Q-Learning Update

def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    """
    One Q-learning update.
    """

    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state])

    Q[state, action] += alpha * (target - Q[state, action])

# 4) Run Dyna-Q or Dyna-Q+

def run_dyna_algorithm(
    env,
    algorithm="dyna_q",
    planning_steps=10,
    episodes=600,
    alpha=0.3,
    gamma=0.99,
    epsilon=0.1,
    kappa=0.001,
    seed=42
):

    rng = np.random.default_rng(seed)

    Q = np.zeros((env.n_states, env.n_actions))

    model = {}
    model_keys = []

    last_tried = np.zeros((env.n_states, env.n_actions))

    time_step = 0

    episode_returns = []
    episode_lengths = []
    model_corrections = []

    for episode in range(episodes):

        env.set_episode(episode)

        state, info = env.reset()

        total_reward = 0.0
        steps = 0
        corrections_this_episode = 0

        while True:

            time_step += 1

            action = epsilon_greedy_action(
                Q=Q,
                state=state,
                epsilon=epsilon,
                n_actions=env.n_actions,
                rng=rng
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            done_for_update = terminated or truncated
            done_for_model = terminated

            q_learning_update(
                Q=Q,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done_for_update,
                alpha=alpha,
                gamma=gamma
            )
            old_model_value = model.get((state, action), None)
            new_model_value = (next_state, reward, done_for_model)

            if episode >= env.change_episode and old_model_value is not None:
                if old_model_value != new_model_value:
                    corrections_this_episode += 1

            if (state, action) not in model:
                model_keys.append((state, action))

            model[(state, action)] = new_model_value

            last_tried[state, action] = time_step

    
            if algorithm == "dyna_q_plus":
                for a in range(env.n_actions):
                    if (state, a) not in model:
                        model[(state, a)] = (state, 0.0, False)
                        model_keys.append((state, a))

            for _ in range(planning_steps):

                random_index = rng.integers(len(model_keys))
                simulated_state, simulated_action = model_keys[random_index]

                simulated_next_state, simulated_reward, simulated_done = model[
                    (simulated_state, simulated_action)
                ]

                if algorithm == "dyna_q_plus":
                    tau = time_step - last_tried[simulated_state, simulated_action]
                    bonus = kappa * np.sqrt(tau)
                    simulated_reward = simulated_reward + bonus

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

            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        model_corrections.append(corrections_this_episode)

    return Q, episode_returns, episode_lengths, model_corrections

# 5) Moving Average

def moving_average(data, window=20):
    """
    Compute moving average for smoother plots.
    """

    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window) / window, mode="valid")


def moving_average_x(data, window=20):
    """
    X-axis values for moving average plots.
    """

    if len(data) < window:
        return np.arange(len(data))

    return np.arange(window - 1, len(data))

# 6) Print Final Policy

def print_policy(env, Q):
    """
    Print final greedy policy.
    """

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

            elif state_tuple in env.walls:
                row_symbols.append("W")

            else:
                state_id = env.state_to_id(state_tuple)
                best_action = np.argmax(Q[state_id])
                row_symbols.append(arrows[best_action])

        print(" ".join(f"{symbol:>2}" for symbol in row_symbols))

# 7) Main Experiment

def main():

    rows = 5
    cols = 20
    max_steps = 1000

    episodes = 600
    change_episode = 250

    planning_steps = 10
    num_runs = 10

    alpha = 0.3
    gamma = 0.99
    epsilon = 0.1
    kappa = 0.001

    window = 20

    algorithms = ["dyna_q", "dyna_q_plus"]

    print("=" * 70)
    print(f"Changing Cliff Walking Environment: {rows}x{cols}")
    print(f"Algorithms: Dyna-Q and Dyna-Q+")
    print(f"Planning steps: {planning_steps}")
    print(f"Episodes: {episodes}")
    print(f"Environment changes at episode: {change_episode}")
    print(f"Runs per algorithm: {num_runs}")
    print("=" * 70)

    results = {}

    for algorithm in algorithms:

        print("\n" + "=" * 70)
        print(f"Running {algorithm}")
        print("=" * 70)

        all_returns = []
        all_lengths = []
        all_corrections = []

        final_Q = None
        final_env = None

        for run in range(num_runs):

            env = ChangingCliffWalkingEnv(
                rows=rows,
                cols=cols,
                max_steps=max_steps,
                change_episode=change_episode
            )

            Q, returns, lengths, corrections = run_dyna_algorithm(
                env=env,
                algorithm=algorithm,
                planning_steps=planning_steps,
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                kappa=kappa,
                seed=run
            )

            all_returns.append(returns)
            all_lengths.append(lengths)
            all_corrections.append(corrections)

            final_Q = Q
            final_env = env

        mean_returns = np.mean(all_returns, axis=0)
        mean_lengths = np.mean(all_lengths, axis=0)
        mean_corrections = np.mean(all_corrections, axis=0)

        results[algorithm] = {
            "Q": final_Q,
            "env": final_env,
            "mean_returns": mean_returns,
            "mean_lengths": mean_lengths,
            "mean_corrections": mean_corrections
        }

        avg_steps_before = np.mean(mean_lengths[change_episode - 50:change_episode])
        avg_steps_after = np.mean(mean_lengths[-50:])

        avg_return_before = np.mean(mean_returns[change_episode - 50:change_episode])
        avg_return_after = np.mean(mean_returns[-50:])

        total_corrections_after = np.sum(mean_corrections[change_episode:])

        print(f"Algorithm: {algorithm}")
        print(f"Average steps before change: {avg_steps_before:.3f}")
        print(f"Average steps after change:  {avg_steps_after:.3f}")
        print(f"Average return before change: {avg_return_before:.3f}")
        print(f"Average return after change:  {avg_return_after:.3f}")
        print(f"Average model corrections after change: {total_corrections_after:.3f}")

        final_env.set_episode(episodes)

        print("\nFinal policy after environment change:")
        print_policy(final_env, final_Q)

    # 8) Plot Average Number of Steps


    plt.figure(figsize=(11, 5))

    for algorithm in algorithms:

        lengths = results[algorithm]["mean_lengths"]
        y = moving_average(lengths, window=window)
        x = moving_average_x(lengths, window=window)

        plt.plot(x, y, label=algorithm)

    plt.axvline(
        x=change_episode,
        linestyle="--",
        label="Environment changed"
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Number of Steps")
    plt.title("Dyna-Q vs Dyna-Q+ in Changing Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

    # 9) Plot Average Return

    plt.figure(figsize=(11, 5))

    for algorithm in algorithms:

        returns = results[algorithm]["mean_returns"]
        y = moving_average(returns, window=window)
        x = moving_average_x(returns, window=window)

        plt.plot(x, y, label=algorithm)

    plt.axvline(
        x=change_episode,
        linestyle="--",
        label="Environment changed"
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Returns: Dyna-Q vs Dyna-Q+ in Changing Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

    # 10) Plot Average Model Corrections

    plt.figure(figsize=(11, 5))

    for algorithm in algorithms:

        corrections = results[algorithm]["mean_corrections"]
        y = moving_average(corrections, window=window)
        x = moving_average_x(corrections, window=window)

        plt.plot(x, y, label=algorithm)

    plt.axvline(
        x=change_episode,
        linestyle="--",
        label="Environment changed"
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Model Corrections")
    plt.title("Model Corrections After Environment Change")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

# 11) Run Program


if __name__ == "__main__":
    main()