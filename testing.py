import gymnasium as gym
import pickle
import numpy as np
from discretize import *
# 1. Load your Q-table
with open("q-table-2026-02-03_17-24-49", "rb") as f:
    q_table = pickle.load(f)

# 2. Setup the environment for visualization
env = gym.make("LunarLander-v3", render_mode="human")
discretizer = StateDiscretizer(env) # Use your existing class

# 3. Run evaluation episodes
for episode in range(10):
    state, _ = env.reset()
    state = discretizer.discretize(state)
    total_reward = 0
    done = False
    
    while not done:
        # Exploit: Always pick the action with the highest Q-value
        action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Move to next state
        state = discretizer.discretize(next_state)
        total_reward += reward
        
    print(f"Episode {episode + 1} Total Reward: {total_reward}")

env.close()