import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

np.set_printoptions(precision=3, suppress=True)

# 1) Cliff Walking Environment - Gymnasium Style

class CliffWalkingEnv:
    def __init__(self, width=20, height=5, max_steps=1000):
        self.width = width
        self.height = height
        self.max_steps = max_steps

        self.n_states = width * height
        self.n_actions = 4

        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)

        self.cliff = set()
        for c in range(1, width - 1):
            self.cliff.add((height - 1, c))

        self.moves = {
            0: (-1, 0),   
            1: (1, 0),    
            2: (0, -1),  
            3: (0, 1)     
        }

        self.state = None
        self.steps = 0

    def encode(self, row, col):
        return row * self.width + col

    def decode(self, state):
        return divmod(state, self.width)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.state = self.start
        self.steps = 0

        state_id = self.encode(*self.state)
        info = {}

        return state_id, info

    def step(self, action):
        if action not in self.moves:
            raise ValueError("Invalid action.")

        self.steps += 1

        row, col = self.state
        d_row, d_col = self.moves[action]

        new_row = row + d_row
        new_col = col + d_col

        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))

        new_state = (new_row, new_col)

        terminated = False
        truncated = False

        if new_state == self.goal:
            reward = -1.0
            terminated = True
            self.state = new_state

        elif new_state in self.cliff:
            reward = -100.0
            terminated = False
            self.state = self.start

        else:
            reward = -1.0
            terminated = False
            self.state = new_state

        if self.steps >= self.max_steps:
            truncated = True

        next_state_id = self.encode(*self.state)
        info = {}

        return next_state_id, reward, terminated, truncated, info

    def render_policy(self, Q):

        action_symbols = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→"
        }

        for r in range(self.height):
            row_symbols = []

            for c in range(self.width):
                state = (r, c)
                state_id = self.encode(r, c)

                if state == self.start:
                    row_symbols.append("S")

                elif state == self.goal:
                    row_symbols.append("G")

                elif state in self.cliff:
                    row_symbols.append("X")

                else:
                    best_action = np.argmax(Q[state_id])
                    row_symbols.append(action_symbols[best_action])

            print(" ".join(f"{x:>2}" for x in row_symbols))

    def print_state_values(self, Q):
        V = np.max(Q, axis=1)

        for r in range(self.height):
            row_values = []

            for c in range(self.width):
                state = (r, c)
                state_id = self.encode(r, c)

                if state == self.goal:
                    row_values.append("   G   ")

                elif state in self.cliff:
                    row_values.append("   X   ")

                else:
                    row_values.append(f"{V[state_id]:7.3f}")

            print(" ".join(row_values))

# 2) Epsilon-Greedy Policy

def epsilon_greedy(Q, state, epsilon, n_actions, rng):


    if rng.random() < epsilon:
        return rng.integers(n_actions)

    q_values = Q[state]
    max_q = np.max(q_values)

    best_actions = np.flatnonzero(q_values == max_q)

    return rng.choice(best_actions)

# 3) n-step SARSA Algorithm

def n_step_sarsa(
    env,
    n,
    num_episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):

    rng = np.random.default_rng(seed)

    Q = np.zeros((env.n_states, env.n_actions))

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):

        state, info = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env.n_actions, rng)

        states = [state]
        actions = [action]
        rewards = [0.0]

        T = float("inf")
        t = 0

        total_reward = 0.0

        while True:

            if t < T:

                next_state, reward, terminated, truncated, info = env.step(actions[t])
                done = terminated or truncated

                states.append(next_state)
                rewards.append(reward)

                total_reward += reward

                if done:
                    T = t + 1

                else:
                    next_action = epsilon_greedy(Q, next_state, epsilon, env.n_actions, rng)
                    actions.append(next_action)

            tau = t - n + 1

            if tau >= 0:

                G = 0.0

                upper_limit = min(tau + n, T)

                for i in range(tau + 1, int(upper_limit) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]

                if tau + n < T:
                    G += (gamma ** n) * Q[states[tau + n], actions[tau + n]]

                s_tau = states[tau]
                a_tau = actions[tau]

                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau == T - 1:
                break

            t += 1

        episode_returns.append(total_reward)
        episode_lengths.append(env.steps)

    return Q, episode_returns, episode_lengths

# 4) Moving Average
def moving_average(data, window=20):
    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window) / window, mode="valid")

# 5) Main Experiment
def main():

    width = 20
    height = 5
    max_steps = 1000

    n_values = [1, 5, 10]

    num_episodes = 500
    num_runs = 10

    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1

    print("=" * 70)
    print(f"Cliff Walking Environment: {height}x{width}")
    print(f"n values: {n_values}")
    print(f"Episodes: {num_episodes}")
    print(f"Runs per n: {num_runs}")
    print("=" * 70)

    results = {}

    for n in n_values:

        print("\n" + "=" * 70)
        print(f"Running n-step SARSA for n = {n}")
        print("=" * 70)

        all_returns = []
        all_lengths = []

        final_Q = None

        for run in range(num_runs):

            env = CliffWalkingEnv(width=width, height=height, max_steps=max_steps)

            Q, returns, lengths = n_step_sarsa(
                env=env,
                n=n,
                num_episodes=num_episodes,
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
        avg_length_last_100 = np.mean(mean_lengths[-100:])

        print(f"\nn = {n}")
        print(f"Average return in last 100 episodes: {avg_return_last_100:.3f}")
        print(f"Average steps in last 100 episodes: {avg_length_last_100:.3f}")

        print("\nFinal learned policy:")
        env.render_policy(final_Q)

        print("\nState values V(s) = max_a Q(s,a):")
        env.print_state_values(final_Q)

    # 6) Plot Average Return

    plt.figure(figsize=(11, 5))

    window = 20

    for n in n_values:

        returns = results[n]["mean_returns"]
        smoothed_returns = moving_average(returns, window=window)

        x = np.arange(len(smoothed_returns)) + window

        plt.plot(x, smoothed_returns, label=f"n-step SARSA, n = {n}")

    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Average Return for n-step SARSA on Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

    # 7) Plot Average Number of Steps

    plt.figure(figsize=(11, 5))

    for n in n_values:

        lengths = results[n]["mean_lengths"]
        smoothed_lengths = moving_average(lengths, window=window)

        x = np.arange(len(smoothed_lengths)) + window

        plt.plot(x, smoothed_lengths, label=f"n-step SARSA, n = {n}")

    plt.xlabel("Episode")
    plt.ylabel("Average Number of Steps")
    plt.title("Average Number of Steps for n-step SARSA on Cliff Walking 5x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.show()

# 8) Run Program

if __name__ == "__main__":
    main()