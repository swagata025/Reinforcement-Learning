import gymnasium as gym
import numpy as np
import pickle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

class CEMAgent:
    def __init__(self, n_inputs, n_actions):
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.param_dim = n_inputs * n_actions + n_actions

    def get_action(self, state, weights_vector):
        w_size = self.n_inputs * self.n_actions
        W = weights_vector[:w_size].reshape(self.n_inputs, self.n_actions)
        b = weights_vector[w_size:]
        logits = state @ W + b
        return np.argmax(logits)

def test_model(model_name="cem_weights_best.pkl", episodes=5):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        # find latest
        files = [f for f in os.listdir(MODELS_DIR) if f.startswith("cem_weights") and f.endswith(".pkl")]
        if not files:
            print("No model found.")
            return
        model_name = sorted(files)[-1] # primitive sort, usually works with updates
        model_path = os.path.join(MODELS_DIR, model_name)

    print(f"Loading weights from {model_path}")
    with open(model_path, "rb") as f:
        best_weights = pickle.load(f)

    env = gym.make("LunarLander-v3", render_mode="human")
    agent = CEMAgent(8, 4)

    total_rewards = []
    
    for i in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.get_action(state, best_weights)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        print(f"Episode {i+1}: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()
    print(f"Average Score: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    test_model()
