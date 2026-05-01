import numpy as np
import gymnasium as gym

def train_agent():
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Q-table
    Q = np.zeros((state_size, action_size))

    # Hyperparameters
    alpha = 0.8
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    episodes = 1000
    max_steps = 100

    rewards_per_episode = []

    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_rewards = 0

        for step in range(max_steps):

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )

            state = new_state
            total_rewards += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_rewards)

    success_rate = sum(rewards_per_episode[-100:]) / 100

    return Q, rewards_per_episode, success_rate

