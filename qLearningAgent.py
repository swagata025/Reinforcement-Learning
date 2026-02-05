import numpy as np
import random
import pickle
class QLearningAgent:
    def __init__(self, discretizer, action_size, learning_rate=0.10, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.discretizer = discretizer
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_lr = 0.01 
        self.lr_decay = 0.9999
        
        # Initialize Q-Table with zeros
        # Shape: (dim1, dim2, ..., dim8, action_size)
        self.q_table = np.zeros(self.discretizer.get_state_space_shape() + (action_size,))

    def choose_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1) # Explore
        return np.argmax(self.q_table[state]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        # Current Q-value
        old_value = self.q_table[state + (action,)]
        
        # Max future Q-value
        next_max = np.max(self.q_table[next_state])
        
        # Update Q-value
        new_value = old_value + self.lr * (reward + self.gamma * next_max * (1 - done) - old_value)
        self.q_table[state + (action,)] = new_value
        
        # Decay epsilon
        if done:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            if self.epsilon < 0.15: self.lr = max(self.min_lr, self.lr * self.lr_decay)
    
    def save(self, filename="q-table.pkl"):
        """Saves the Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load(self, filename="q-table.pkl"):
        """Loads a Q-table from a file."""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")