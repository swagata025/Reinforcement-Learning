import gymnasium as gym
import numpy as np
import pickle
import os
from state_discretizer import StateDiscretizer

# Path to the trained model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
# To load a specific training session, change this filename.
# Example: "q_table_21-01-26_14-35-00.pkl"
# Default: "q_table.pkl" (This is always the most recently trained agent)
MODEL_FILENAME = "q_table.pkl" 
# -----------------------------------------------------------------

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

def test_model(episodes=5):
    """
    Loads the trained Q-table and runs the agent in human render mode
    to visualize how well it plays the game.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No trained model found at {MODEL_PATH}")
        print("Please run 'lunar_lander_qlearning.py' to train the agent first.")
        return

    print(f"Loading trained model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        q_table = pickle.load(f)

    env = gym.make("LunarLander-v3", render_mode="human")
    
    discretizer = StateDiscretizer(env)
    
    print(f"\nVisualizing {episodes} episodes...")
    print("-----------------------------------")
    
    total_rewards_all_episodes = []

    for ep in range(episodes):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)
        
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (terminated or truncated):
            # Exploit: Always choose the best action from the Q-table
            action = np.argmax(q_table[discrete_state])
            
            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update state for next step
            discrete_state = discretizer.discretize(next_state)
            total_reward += reward
            step += 1
            
        print(f"Episode {ep + 1}: Total Reward: {total_reward:.2f} (Steps: {step})")
        total_rewards_all_episodes.append(total_reward)
    
    env.close()
    avg_reward = sum(total_rewards_all_episodes) / episodes
    print("-----------------------------------")
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print("Visualization complete.")

if __name__ == "__main__":
    test_model(episodes=8)
