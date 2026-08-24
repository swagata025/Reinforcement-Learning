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
# Default: "rewards_history.npy" (This is always the most recent run)
REWARDS_FILENAME = "rewards_history.npy"
# -----------------------------------------------------------------

REWARDS_PATH = os.path.join(DATA_DIR, REWARDS_FILENAME)

def plot_learning_curve(rewards_file=REWARDS_PATH, save_dir=None, timestamp=None, show_plot=True):
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
    
    plt.title("LunarLander-v3 Training Progress (SARSA)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        # Timestamp format: DD-MM-YY_HH-MM-SS (Safe for Windows filenames)
        timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    plot_filename = f"learning_curve_{timestamp}.png"
    
    # Use default output dir if not provided
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "plots")
        
    output_path = os.path.join(save_dir, plot_filename)
    plt.savefig(output_path)
    print(f"Plot saved as {plot_filename}")
    
    if show_plot:
        plt.show()    
    else:
        plt.close() # Close figure to free memory if not showing

if __name__ == "__main__":
    plot_learning_curve()
