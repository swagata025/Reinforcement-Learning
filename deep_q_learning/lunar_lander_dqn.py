import gymnasium as gym
import numpy as np
import random
from collections import deque
import os
import datetime
import pickle
import copy
from sklearn.neural_network import MLPRegressor
import warnings

# Suppress warnings about learning from scratch on partial_fit
warnings.filterwarnings("ignore")

# Hyperparameters
gamma = 0.99
batch_size = 128
lr = 5e-4
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.996
buffer_size = 200000
target_update_freq = 15 
num_episodes = 2000

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Fixed file names
FIXED_MODEL_PATH = os.path.join(MODELS_DIR, "dqn_lunar_lander.pkl")
FIXED_REWARDS_PATH = os.path.join(DATA_DIR, "rewards_history.npy")

class SklearnDQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # MLPRegressor config for RL
        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation='relu',
            solver='adam',
            learning_rate='constant',
            learning_rate_init=lr,
            warm_start=True,   # Keep weights between fits
            max_iter=1,        # 1 iteration per fit/partial_fit call
            batch_size=batch_size,
            random_state=42
        )
        
        # Initialize model with dummy data to set architecture
        # We need to partial_fit once to initialize weights
        dummy_state = np.zeros((1, state_dim))
        dummy_target = np.zeros((1, action_dim))
        self.model.fit(dummy_state, dummy_target)

    def predict(self, state):
        # State: (n_samples, state_dim)
        return self.model.predict(state)

    def fit(self, X, y):
        self.model.partial_fit(X, y)
        
    def copy_weights_from(self, other_model):
        # Sklearn doesn't have load_state_dict/state_dict logic like PyTorch
        # We model deepcopy the internal coefficients
        self.model.coefs_ = copy.deepcopy(other_model.model.coefs_)
        self.model.intercepts_ = copy.deepcopy(other_model.model.intercepts_)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

def train_dqn():
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize Networks
    policy_net = SklearnDQN(state_dim, action_dim)
    target_net = SklearnDQN(state_dim, action_dim)
    target_net.copy_weights_from(policy_net)

    replay_buffer = ReplayBuffer(buffer_size)

    rewards_history = []
    epsilon = epsilon_start

    print("Starting DQN Training (sklearn)...")

    # Import visualization tool
    from visualization import plot_learning_curve

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = policy_net.predict(state.reshape(1, -1))
                action = np.argmax(q_values[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Training Step
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Predict Q(s', a') from target network
                next_qs = target_net.predict(next_states)
                max_next_qs = np.max(next_qs, axis=1)
                
                # TD Target
                targets = rewards + gamma * max_next_qs * (1 - dones)

                # Get current Q(s) predictions to update
                # We want to update ONLY the taken action's Q-value to 'targets'
                # The other actions should act as "no error" (target = current prediction)
                current_qs = policy_net.predict(states)
                
                # Update specific action indices
                target_qs_for_training = current_qs.copy()
                for i in range(batch_size):
                    target_qs_for_training[i, actions[i]] = targets[i]
                
                # Train
                policy_net.fit(states, target_qs_for_training)

        # Update Target Network
        if episode % target_update_freq == 0:
            target_net.copy_weights_from(policy_net)

        # Decay Epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)

        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

        # Periodic Save
        if (episode + 1) % 10 == 0:
            with open(FIXED_MODEL_PATH, "wb") as f:
                pickle.dump(policy_net.model, f)
            np.save(FIXED_REWARDS_PATH, np.array(rewards_history))

    # Final Save (Fixed)
    with open(FIXED_MODEL_PATH, "wb") as f:
        pickle.dump(policy_net.model, f)
    np.save(FIXED_REWARDS_PATH, np.array(rewards_history))

    # Final Save (Timestamped History)
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    timestamped_model_path = os.path.join(MODELS_DIR, f"dqn_lunar_lander_{timestamp}.pkl")
    timestamped_rewards_path = os.path.join(DATA_DIR, f"rewards_history_{timestamp}.npy")

    with open(timestamped_model_path, "wb") as f:
        pickle.dump(policy_net.model, f)
    np.save(timestamped_rewards_path, np.array(rewards_history))
    
    print(f"Training Complete.")
    print(f"Model saved to:\n  - {FIXED_MODEL_PATH} (Latest)\n  - {timestamped_model_path} (History)")
    print(f"Rewards saved to:\n  - {FIXED_REWARDS_PATH} (Latest)\n  - {timestamped_rewards_path} (History)")

    # Visualization
    try:
        plot_learning_curve(rewards_file=FIXED_REWARDS_PATH, timestamp=timestamp)
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    train_dqn()
