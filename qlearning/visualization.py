import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_learning_curve(rewards_file=os.path.join(SCRIPT_DIR, "data", "rewards_history.npy")):
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
    
    plt.savefig(os.path.join(SCRIPT_DIR, "plots", "learning_curve.png"))
    print("Plot saved as learning_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()
