import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
# To plot a specific training session, change this filename.
# Example: "rewards_history_21-01-26_14-35-00.npy"
# Default: "rewards_history.npy" (This is always the most recent run)
REWARDS_FILENAME = "rewards_history.npy"
# -----------------------------------------------------------------

REWARDS_PATH = os.path.join(DATA_DIR, REWARDS_FILENAME)

def plot_learning_curve(rewards_file=REWARDS_PATH):
    if not os.path.exists(rewards_file):
        print(f"File {rewards_file} not found. Run training first.")
        return

    rewards = np.load(rewards_file)
    
    # Calculate Running Average
    window_size = 100
    running_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window_size)
        running_avg.append(np.mean(rewards[start:i+1]))
        
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.3, color='blue')
    plt.plot(running_avg, label=f"{window_size}-Episode Moving Average", color='red', linewidth=2)
    
    plt.title("LunarLander-v3 Training Progress (Tabular Q-Learning)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # Timestamp format: DD-MM-YY_HH-MM-SS (Safe for Windows filenames)
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    plot_filename = f"learning_curve_{timestamp}.png"
    plt.savefig(os.path.join(SCRIPT_DIR, "plots", plot_filename))
    print(f"Plot saved as {plot_filename}")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()
