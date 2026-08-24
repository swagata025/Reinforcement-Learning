import gymnasium as gym
import numpy as np
import os
import pickle
import sys

def test_dqn(episodes=5):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    MODEL_PATH = os.path.join(MODELS_DIR, "dqn_lunar_lander.pkl")

    # If default model doesn't exist, try to find the latest sklearn_dqn_*.pkl
    if not os.path.exists(MODEL_PATH):
        print(f"Default model {MODEL_PATH} not found. Searching for latest...")
        if os.path.exists(MODELS_DIR):
            files = [f for f in os.listdir(MODELS_DIR) if f.startswith("sklearn_dqn") and f.endswith(".pkl")]
            if files:
                latest = max(files, key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)))
                MODEL_PATH = os.path.join(MODELS_DIR, latest)
                print(f"Found latest model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train the agent first.")
        return

    env = gym.make("LunarLander-v3", render_mode="human")
    
    # Load Sklearn Model
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Testing Scikit-Learn DQN Model for {episodes} episodes...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Predict action
            q_values = model.predict(state.reshape(1, -1))
            action = np.argmax(q_values[0])

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_dqn()
