# Warehouse Robot Navigation with Deep Q-Learning

A reinforcement learning project in which a warehouse robot learns to navigate a custom grid environment, collect a package, and deliver it to a target location using Deep Q-Learning (DQN).

The environment is implemented entirely from scratch without Gymnasium or any ready-made reinforcement learning environment.

## Environment

The robot operates in a 5×5 warehouse grid containing:

- Start position
- Package location
- Delivery goal
- Static obstacles
- Energy constraints
- Collision penalties

The agent must first collect the package and then navigate to the delivery target while minimizing unnecessary movements, collisions, and energy consumption.

## State Representation

The DQN receives the following normalized state:

`[row, column, has_package, energy]`

## Actions

- Left
- Down
- Right
- Up

## Reinforcement Learning Method

The agent is trained using Deep Q-Learning with:

- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Bellman Updates
- Huber Loss
- Gradient Clipping
- Hard and Soft Target Updates

## Evaluation

The project includes:

- Training reward analysis
- Success-rate tracking
- Energy-efficiency analysis
- Obstacle-hit statistics
- Episode-length analysis
- DQN loss monitoring
- Learned policy visualization
- Greedy policy execution
- Pygame-based environment visualization
