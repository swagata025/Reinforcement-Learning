import gymnasium as gym
import numpy as np
import pickle
import os
from state_discretizer import StateDiscretizer
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.15      # Slightly increased to learn faster from good events
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.998     # Slower decay to explore the new state space thoroughly
EPSILON_MIN = 0.02        # Keep a bit more randomness
EPISODES = 12000           # Give it enough time to converge
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "q_table.pkl")

def train_agent():
    # Create Environment
    env = gym.make("LunarLander-v3")
    
    # Initialize Discretizer
    discretizer = StateDiscretizer(env)
    
    # Initialize Q-Table
    # We use a map (dictionary) or a large numpy array. 
    # Since the state space is fixed, let's use a numpy array for speed if memory allows.
    state_shape = discretizer.get_state_space_shape()
    action_size = env.action_space.n
    
    print(f"State Space Shape: {state_shape}")
    print(f"Action Space Size: {action_size}")
    
    # Initialize Q-table with zeros
    # Note: If memory error occurs, switch to dictionary: q_table = {}
    try:
        q_table = np.zeros(state_shape + (action_size,))
        print("Q-table initialized successfully.")
    except MemoryError:
        print("State space too large for array, check binning strategy.")
        return

    # Training Metrics
    rewards_history = []
    epsilon = EPSILON_START
    
    print("Starting Training...")
    
    for episode in range(EPISODES):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)
        
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Epsilon-Greedy Action Selection
            if np.random.random() < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(q_table[discrete_state]) # Exploit
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            next_discrete_state = discretizer.discretize(next_state)
            
            # Q-Learning Update Rule
            best_next_action = np.argmax(q_table[next_discrete_state])
            td_target = reward + DISCOUNT_FACTOR * q_table[next_discrete_state][best_next_action]
            td_error = td_target - q_table[discrete_state][action]
            
            q_table[discrete_state][action] += LEARNING_RATE * td_error
            
            # Update state
            discrete_state = next_discrete_state
            total_reward += reward
            
        # Decay Epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode: {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Save Agent
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {SAVE_PATH}")
    
    # Save Metrics for Visualization
    np.save(os.path.join(SCRIPT_DIR, "data", "rewards_history.npy"), np.array(rewards_history))
    
    return q_table, rewards_history

def test_agent(episodes=5):
    if not os.path.exists(SAVE_PATH):
        print("No trained agent found. Run training first.")
        return

    with open(SAVE_PATH, "rb") as f:
        q_table = pickle.load(f)

    # Render mode for testing
    env = gym.make("LunarLander-v3", render_mode="human")
    discretizer = StateDiscretizer(env)
    
    print("\nTesting Trained Agent...")
    
    for ep in range(episodes):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            action = np.argmax(q_table[discrete_state])
            next_state, reward, terminated, truncated, info = env.step(action)
            discrete_state = discretizer.discretize(next_state)
            total_reward += reward
            
        print(f"Test Episode {ep + 1}: Total Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # You can toggle these
    # Train
    train_agent()
    # Test
    # test_agent()
