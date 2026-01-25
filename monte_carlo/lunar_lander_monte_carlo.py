import datetime
import os
import pickle
import sys

import gymnasium as gym
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from monte_carlo.state_discretizer_mc import StateDiscretizerMC

# Hyperparameters
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99985 # Slower decay for longer training
EPSILON_MIN = 0.01
EPISODES = 50000       # Significantly more episodes for convergence
MC_ALPHA = 0.01        # Base learning rate (unused if using sample average)
EVERY_VISIT = True
SHAPING_ANGLE = 0.3
SHAPING_VEL = 0.3      # Increased velocity penalty
SHAPING_DIST = 1.0     # Stronger centering penalty
SHAPING_LEG_BONUS = 0.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
SAVE_PATH = os.path.join(MODELS_DIR, "mc_q_table.pkl")


def epsilon_greedy_action(q_table, discrete_state, action_size, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(action_size)
    return int(np.argmax(q_table[discrete_state]))


def train_agent():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = gym.make("LunarLander-v3")
    discretizer = StateDiscretizerMC(env)

    state_shape = discretizer.get_state_space_shape()
    action_size = env.action_space.n

    print(f"State Space Shape: {state_shape}")
    print(f"Action Space Size: {action_size}")

    try:
        # Optimistic Initialization to encourage exploration
        q_table = np.full(state_shape + (action_size,), 50.0)
        returns_count = np.zeros(state_shape + (action_size,))
        print("Q-table initialized successfully with optimistic values.")
    except MemoryError:
        print("State space too large for array, check binning strategy.")
        return

    rewards_history = []
    epsilon = EPSILON_START

    print("Starting Monte Carlo Training...")

    best_avg_reward = -float('inf')

    for episode in range(EPISODES):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)

        terminated = False
        truncated = False
        episode_steps = []
        total_reward = 0.0

        while not (terminated or truncated):
            action = epsilon_greedy_action(q_table, discrete_state, action_size, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_discrete_state = discretizer.discretize(next_state)

            # Light reward shaping for stability
            pos_x = next_state[0]
            angle = next_state[4]
            vel_x = next_state[2]
            vel_y = next_state[3]
            leg_left = next_state[6]
            leg_right = next_state[7]

            shaping = -SHAPING_ANGLE * abs(angle) - SHAPING_VEL * (abs(vel_x) + abs(vel_y))
            shaping -= SHAPING_DIST * abs(pos_x)  # Penalize being off-center
            shaping += SHAPING_LEG_BONUS * (leg_left + leg_right)

            episode_steps.append((discrete_state, action, reward + shaping))
            total_reward += reward
            discrete_state = next_discrete_state

        rewards_history.append(total_reward)

        # Monte Carlo update (backward return computation)
        G = 0.0
        visited = set()
        for t in range(len(episode_steps) - 1, -1, -1):
            state_t, action_t, reward_t = episode_steps[t]
            G = DISCOUNT_FACTOR * G + reward_t
            sa_key = (state_t, action_t)
            if EVERY_VISIT or sa_key not in visited:
                if not EVERY_VISIT:
                    visited.add(sa_key)
                returns_count[state_t][action_t] += 1
                
                # Sample Average Update (True Monte Carlo)
                # Alpha is effectively 1/N
                N = returns_count[state_t][action_t]
                alpha = 1.0 / N
                q_table[state_t][action_t] += alpha * (G - q_table[state_t][action_t])

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode: {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, "
                f"Epsilon: {epsilon:.4f}"
            )
            
            # Save Best Model Logic
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                with open(SAVE_PATH, "wb") as f:
                    pickle.dump(q_table, f)
                print(f"  > New Best Average Reward! Model saved to {SAVE_PATH}")

    # Save Agent
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    model_filename = f"mc_q_table_{timestamp}.pkl"
    save_path_timestamp = os.path.join(MODELS_DIR, model_filename)

    with open(save_path_timestamp, "wb") as f:
        pickle.dump(q_table, f)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(q_table, f)

    print(f"Q-table saved to {save_path_timestamp} (and updated {SAVE_PATH})")

    rewards_filename = f"mc_rewards_history_{timestamp}.npy"
    rewards_path_timestamp = os.path.join(DATA_DIR, rewards_filename)
    np.save(rewards_path_timestamp, np.array(rewards_history))

    np.save(os.path.join(DATA_DIR, "mc_rewards_history.npy"), np.array(rewards_history))
    print(f"Rewards history saved to {rewards_path_timestamp}")

    try:
        from monte_carlo.visualization import plot_learning_curve

        plot_learning_curve(rewards_file=rewards_path_timestamp, show=False)
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    return q_table, rewards_history


def test_agent(episodes=5):
    if not os.path.exists(SAVE_PATH):
        print("No trained agent found. Run training first.")
        return

    with open(SAVE_PATH, "rb") as f:
        q_table = pickle.load(f)

    env = gym.make("LunarLander-v3", render_mode="human")
    discretizer = StateDiscretizerMC(env)

    print("\nTesting Trained Agent...")

    for ep in range(episodes):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = int(np.argmax(q_table[discrete_state]))
            next_state, reward, terminated, truncated, info = env.step(action)
            discrete_state = discretizer.discretize(next_state)
            total_reward += reward

        print(f"Test Episode {ep + 1}: Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    train_agent()
    # test_agent()
