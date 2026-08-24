import gymnasium as gym
import numpy as np
import pickle
import os
import datetime
import json
from state_discretizer_sarsa import StateDiscretizerSarsa
from visualization import plot_learning_curve

# Hyperparameters
LEARNING_RATE_START = 0.15 # Higher initial learning rate
LEARNING_RATE_MIN = 0.01
LEARNING_RATE_DECAY = 0.99985 # Slower decay
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99965    # Slower decay for larger state space exploration
EPSILON_MIN = 0.01
EPISODES = 15000       # Increased episodes for better convergence
SHAPING_ANGLE = 0.5    # Stronger penalty for tilting
SHAPING_VEL = 0.5      # Stronger penalty for high speed
SHAPING_DIST = 1.5     # Stronger penalty for being off-center
SHAPING_LEG_BONUS = 0.3 # Slightly higher bonus
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "sarsa_q_table.pkl")

def train_agent():
    # Create Environment
    env = gym.make("LunarLander-v3")
    
    # Initialize Discretizer
    discretizer = StateDiscretizerSarsa(env)
    
    # Initialize Q-Table
    state_shape = discretizer.get_state_space_shape()
    action_size = env.action_space.n
    
    print(f"State Space Shape: {state_shape}")
    print(f"Action Space Size: {action_size}")
    
    # Initialize Q-table with zeros
    try:
        q_table = np.zeros(state_shape + (action_size,))
        print("Q-table initialized successfully.")
    except MemoryError:
        print("State space too large for array, check binning strategy.")
        return

    # Training Metrics
    rewards_history = []
    epsilon = EPSILON_START
    learning_rate = LEARNING_RATE_START
    
    print("Starting Training (SARSA)...")
    
    for episode in range(EPISODES):
        state, info = env.reset()
        discrete_state = discretizer.discretize(state)
        
        # SARSA: Select initial action a using policy derived from Q (e.g., epsilon-greedy)
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[discrete_state]) # Exploit

        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            next_discrete_state = discretizer.discretize(next_state)

            # Reward Shaping
            pos_x = next_state[0]
            angle = next_state[4]
            vel_x = next_state[2]
            vel_y = next_state[3]
            leg_left = next_state[6]
            leg_right = next_state[7]

            shaping = -SHAPING_ANGLE * abs(angle) - SHAPING_VEL * (abs(vel_x) + abs(vel_y))
            shaping -= SHAPING_DIST * abs(pos_x)
            shaping += SHAPING_LEG_BONUS * (leg_left + leg_right)
            
            shaped_reward = reward + shaping
            
            # SARSA: Select next action a' using policy derived from Q (e.g., epsilon-greedy)
            if np.random.random() < epsilon:
                next_action = env.action_space.sample() # Explore
            else:
                next_action = np.argmax(q_table[next_discrete_state]) # Exploit
            
            # SARSA Update Rule: Q(s,a) <-- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
            # We use shaped_reward for the update to guide learning
            td_target = shaped_reward + DISCOUNT_FACTOR * q_table[next_discrete_state][next_action]
            td_error = td_target - q_table[discrete_state][action]
            
            q_table[discrete_state][action] += learning_rate * td_error
            
            # Update state and action for next step
            discrete_state = next_discrete_state
            action = next_action
            
            total_reward += reward
            
        # Decay Epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Decay Learning Rate
        learning_rate = max(LEARNING_RATE_MIN, learning_rate * LEARNING_RATE_DECAY)
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode: {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}, LR: {learning_rate:.4f}")

    # Save Agent
    # Timestamp format: DD-MM-YY_HH-MM-SS (Safe for Windows filenames)
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")

    # Save Hyperparameters
    hyperparams = {
        "LEARNING_RATE_START": LEARNING_RATE_START,
        "LEARNING_RATE_MIN": LEARNING_RATE_MIN,
        "LEARNING_RATE_DECAY": LEARNING_RATE_DECAY,
        "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
        "EPSILON_START": EPSILON_START,
        "EPSILON_DECAY": EPSILON_DECAY,
        "EPSILON_MIN": EPSILON_MIN,
        "EPISODES": EPISODES,
        "SHAPING_ANGLE": SHAPING_ANGLE,
        "SHAPING_VEL": SHAPING_VEL,
        "SHAPING_DIST": SHAPING_DIST,
        "SHAPING_LEG_BONUS": SHAPING_LEG_BONUS
    }
    
    params_filename = f"sarsa_params_{timestamp}.json"
    params_path = os.path.join(SCRIPT_DIR, "models", params_filename)
    with open(params_path, "w") as f:
         json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to {params_path}")
    
    # Save timestamped model
    model_filename = f"sarsa_q_table_{timestamp}.pkl"
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
