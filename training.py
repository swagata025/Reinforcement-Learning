import gymnasium as gym
from discretize import *
from qLearningAgent import *
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------- Setup ----------
now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

env = gym.make("LunarLander-v3")
discretizer = StateDiscretizer(env)
agent = QLearningAgent(discretizer, env.action_space.n, epsilon_decay=0.9994)

save_count = 100
log_interval = 100
num_episodes = 10000

# ---------- Tracking ----------
rewards_per_episode = []
epsilon_history = []

# ---------- Results directory ----------
os.makedirs("results", exist_ok=True)

file = f"q-table-{now_str}"

# ---------- Training ----------
for episode in range(1, num_episodes + 1):

    if episode % save_count == 0:
        agent.save(file)

    raw_state, _ = env.reset()
    state = discretizer.discretize(raw_state)
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_raw_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = discretizer.discretize(next_raw_state)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon_history.append(agent.epsilon)

    # ---------- Logging ----------
    if episode % log_interval == 0:
        avg_reward = np.mean(rewards_per_episode[-log_interval:])
        print(
            f"Episode {episode:5d} | "
            f"Avg Reward (last {log_interval}): {avg_reward:8.2f} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Learning-rate: {agent.lr:.5f}" 
        )

# ---------- Plot ----------
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Episode Reward", alpha=0.6)

# rolling average
window = 100
rolling_avg = np.convolve(
    rewards_per_episode,
    np.ones(window) / window,
    mode="valid"
)
plt.plot(
    range(window - 1, num_episodes),
    rolling_avg,
    label=f"{window}-Episode Avg",
    linewidth=2
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("LunarLander Q-Learning Training")
plt.legend()
plt.grid(True)

plot_path = f"results/reward_vs_episode_{now_str}.png"
plt.savefig(plot_path)
plt.close()

print(f"\n📈 Training plot saved to: {plot_path}")
