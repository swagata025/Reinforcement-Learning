import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Find the latest rewards file automatically
def get_latest_rewards_file():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("rewards_history") and f.endswith(".npy")]
    if not files:
        return None
    # Sort by modification time
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)))
    return os.path.join(DATA_DIR, latest_file)

DEFAULT_REWARDS_FILE = get_latest_rewards_file()

def plot_learning_curve(rewards_file=DEFAULT_REWARDS_FILE, timestamp=None, show_plot=True):
    if rewards_file is None or not os.path.exists(rewards_file):
        print(f"File not found. Ensure training has run.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    rewards = np.load(rewards_file)

    # Calculate Moving Average
    window_size = 50
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.4, color="skyblue")
    plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f"{window_size}-Episode Moving Avg", color="blue", linewidth=2)

    plt.title("Deep Q-Network (DQN) Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.ylim(-400, 400)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    
    save_path = os.path.join(PLOTS_DIR, f"training_curve_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    plot_learning_curve()
