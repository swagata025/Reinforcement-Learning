import os
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'Tabular Q-Learning': os.path.join(script_dir, 'qlearning', 'data', 'rewards_history.npy'),
        'SARSA (On-Policy TD)': os.path.join(script_dir, 'sarsa', 'data', 'rewards_history.npy'),
        'Monte Carlo Control': os.path.join(script_dir, 'monte_carlo', 'data', 'mc_rewards_history.npy'),
        'Cross-Entropy Method': os.path.join(script_dir, 'cross_entropy', 'data', 'rewards_history.npy'),
        'Deep Q-Network (DQN)': os.path.join(script_dir, 'deep_q_learning', 'data', 'rewards_history.npy')
    }
    
    datasets = {}
    for name, path in paths.items():
        if os.path.exists(path):
            datasets[name] = np.load(path)
        else:
            print(f"Warning: {path} not found.")
    return datasets

def calculate_moving_average(data, window=100):
    if len(data) < window:
        return np.convolve(data, np.ones(len(data))/len(data), mode='valid')
    return np.convolve(data, np.ones(window)/window, mode='valid')

def generate_comparison_plot(datasets, output_path):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'Tabular Q-Learning': '#1f77b4',
        'SARSA (On-Policy TD)': '#ff7f0e',
        'Monte Carlo Control': '#2ca02c',
        'Cross-Entropy Method': '#d62728',
        'Deep Q-Network (DQN)': '#9467bd'
    }
    
    final_scores = {}
    
    for name, rewards in datasets.items():
        moving_avg = calculate_moving_average(rewards, window=100)
        episodes = np.arange(1, len(moving_avg) + 1)
        color = colors.get(name, '#333333')
        
        # Line plot
        ax1.plot(episodes, moving_avg, label=name, color=color, linewidth=2.0)
        
        final_scores[name] = np.mean(rewards[-100:])
        
    ax1.axhline(y=200.0, color='red', linestyle='--', linewidth=1.5, label='Environment Solved (200+)')
    ax1.set_title('Learning Curves (100-Episode Moving Average)', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Episodes / Iterations', fontsize=12)
    ax1.set_ylabel('Reward (Score)', fontsize=12)
    ax1.legend(loc='lower right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Bar Chart comparison
    names = list(final_scores.keys())
    scores = [final_scores[n] for n in names]
    bar_colors = [colors.get(n, '#333333') for n in names]
    
    bars = ax2.barh(names, scores, color=bar_colors, alpha=0.85, edgecolor='black', height=0.55)
    ax2.axvline(x=200.0, color='red', linestyle='--', linewidth=1.5, label='Solved Threshold')
    ax2.set_title('Final Evaluation Performance (Mean of Last 100 Episodes)', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Mean Reward', fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        xpos = width + 5 if width >= 0 else width - 25
        ax2.text(xpos, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}', 
                 va='center', ha='left' if width >= 0 else 'right', fontweight='bold', fontsize=10)
                 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Benchmark plot successfully saved to: {output_path}")

def print_benchmark_table(datasets):
    print("=" * 85)
    print(f"{'ALGORITHM BENCHMARK SUMMARY (LunarLander-v2/v3)':^85}")
    print("=" * 85)
    header = f"{'Algorithm':<25} | {'Episodes':<10} | {'Max Reward':<12} | {'Final 100 Avg':<14} | {'Status':<10}"
    print(header)
    print("-" * 85)
    
    for name, rewards in datasets.items():
        episodes = len(rewards)
        max_r = np.max(rewards)
        final_avg = np.mean(rewards[-100:])
        status = "SOLVED" if final_avg >= 200.0 else "TRAINED"
        print(f"{name:<25} | {episodes:<10} | {max_r:<12.1f} | {final_avg:<14.1f} | {status:<10}")
        
    print("=" * 85)

if __name__ == "__main__":
    data = load_data()
    if data:
        print_benchmark_table(data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, 'plots', 'algorithm_comparison.png')
        generate_comparison_plot(data, plot_path)
    else:
        print("No dataset loaded.")
