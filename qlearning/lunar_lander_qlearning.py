import gymnasium as gym
import numpy as np
import pickle
import os
import datetime
from state_discretizer import StateDiscretizer
from visualization import plot_learning_curve
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.1       # Lower learning rate for more stability over long training
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.9996    # Very slow decay: Explores for ~10,000 episodes
EPSILON_MIN = 0.01        # Allow almost full exploitation at the end
EPISODES = 15000          # Long training session
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
    # Timestamp format: DD-MM-YY_HH-MM-SS (Safe for Windows filenames)
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    # Save timestamped model
    model_filename = f"q_table_{timestamp}.pkl"
    save_path_timestamp = os.path.join(SCRIPT_DIR, "models", model_filename)
    with open(save_path_timestamp, "wb") as f:
        pickle.dump(q_table, f)
        
    # Overwrite default for convenience
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {save_path_timestamp} (and updated {SAVE_PATH})")
    
    # Save Metrics for Visualization
    rewards_filename = f"rewards_history_{timestamp}.npy"
    rewards_path_timestamp = os.path.join(SCRIPT_DIR, "data", rewards_filename)
    np.save(rewards_path_timestamp, np.array(rewards_history))
    
    # Overwrite default
    np.save(os.path.join(SCRIPT_DIR, "data", "rewards_history.npy"), np.array(rewards_history))
    print(f"Rewards history saved to {rewards_path_timestamp}")
    
    # Auto-generate plot
    print("Generating learning curve...")
    try:
        plot_learning_curve(rewards_file=rewards_path_timestamp, timestamp=timestamp, show_plot=False)
    except Exception as e:
        print(f"Could not generate plot automatically: {e}")

    return q_table, rewards_history

if __name__ == "__main__":
    train_agent()
