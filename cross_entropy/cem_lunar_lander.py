import gymnasium as gym
import numpy as np
import pickle
import os
import datetime
import json
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------------
# CEM works by maintaining a distribution (mean, deviation) over the POLICY WEIGHTS.
# We sample a batch of "candidate agents", run them, select the best ones ("elites"),
# and update our distribution to look more like the elites.

n_sessions = 100          # Agents to run per generation
percentile = 70           # Top 30% of agents will be used to update the distribution
log_learning_rate = 0.5   # Smooths the mean/std updates (0.0 to 1.0)
n_generations = 200       # How many iterations to run
sigma_decay = 0.99        # Reduce noise over time
min_sigma = 0.1           # Floor for noise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class CEMAgent:
    def __init__(self, n_inputs, n_actions):
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        
        # Policy is a simple linear mapping: State -> Logits using (Weights, Bias)
        # Total parameters = n_inputs * n_actions + n_actions
        self.param_dim = n_inputs * n_actions + n_actions

    def get_action(self, state, weights_vector):
        """
        Computes the action for a given state using specific weights.
        """
        # Unpack the flat weights vector into W and b
        w_size = self.n_inputs * self.n_actions
        W = weights_vector[:w_size].reshape(self.n_inputs, self.n_actions)
        b = weights_vector[w_size:]
        
        # Linear pass
        logits = state @ W + b
        
        # Deterministic action (greedy)
        return np.argmax(logits)

def run_session(env, agent, weights, t_max=1000):
    """
    Runs a single episode ("session") with a specific weight vector.
    Returns total reward.
    """
    total_reward = 0
    state, info = env.reset()
    
    for _ in range(t_max):
        action = agent.get_action(state, weights)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
            
    return total_reward

def train_agent():
    env = gym.make("LunarLander-v3")
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    agent = CEMAgent(n_inputs, n_actions)
    
    print(f"Policy: Linear {n_inputs} -> {n_actions}")
    print(f"Total Parameters to optimize: {agent.param_dim}")
    print(f"Algorithm: Cross-Entropy Method (Policy Search)")
    
    # Initialize Distribution over weights (Gaussian)
    # mean = zeros, std = 1.0 (initially exploring wildly)
    mean = np.zeros(agent.param_dim)
    sigma = np.ones(agent.param_dim) * 10.0 # High initial exploring variance (10.0)

    history = []
    best_mean_reward = -float('inf')

    for i in range(n_generations):
        # 1. Sample 'n_sessions' new weight vectors from our distribution
        # weights_batch shape: (n_sessions, param_dim)
        weights_batch = np.random.randn(n_sessions, agent.param_dim) * sigma + mean # N(mean, sigma)

        # 2. Run episodes for each candidate weight vector
        rewards = []
        for w in weights_batch:
            r = run_session(env, agent, w)
            rewards.append(r)
        
        rewards = np.array(rewards)

        # 3. Select "Elite" candidates (Top 30%)
        # Calculate the threshold score
        reward_threshold = np.percentile(rewards, percentile)
        
        # Get the indices of the weights that scored > threshold
        elite_idxs = rewards >= reward_threshold
        elite_weights = weights_batch[elite_idxs]
        elite_rewards = rewards[elite_idxs]

        # 4. Update the distribution to match the elites
        # We blend old mean/sigma with new mean/sigma for stability
        if len(elite_weights) > 0:
            new_mean = elite_weights.mean(axis=0)
            new_sigma = elite_weights.std(axis=0) + 1e-5 # avoid div by zero
            
            mean = (1 - log_learning_rate) * mean + log_learning_rate * new_mean
            sigma = (1 - log_learning_rate) * sigma + log_learning_rate * new_sigma
            
            # Artificial decay of noise to force convergence eventually
            sigma = np.maximum(sigma * sigma_decay, min_sigma)

        # Logging
        mean_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        threshold_reward = np.mean(elite_rewards)
        
        history.append(mean_reward)
        
        print(f"Gen {i+1}/{n_generations}: Mean Reward: {mean_reward:.2f}, Max: {max_reward:.2f}, Threshold (Elite Avg): {threshold_reward:.2f}, Sigma: {np.mean(sigma):.2f}")

        # Save Checkpoint
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            if mean_reward > 200:
                print(f"  > Solved! (Reward > 200). Saving model.")
            
            # Save the optimal WEIGHTS (the mean of the distribution)
            timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            model_path = os.path.join(MODELS_DIR, f"cem_weights_{timestamp}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(mean, f)
            
            # Overwrite 'latest'
            with open(os.path.join(MODELS_DIR, "cem_weights_best.pkl"), "wb") as f:
                pickle.dump(mean, f)

        if mean_reward > 250:
            print("Converged to high score. Stopping early.")
            break

    # Save training data
    np.save(os.path.join(DATA_DIR, "rewards_history.npy"), np.array(history))
    
    # Plot
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Mean Reward")
    plt.title("CEM Training Progress")
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"))
    print("Training finished.")

if __name__ == "__main__":
    train_agent()
