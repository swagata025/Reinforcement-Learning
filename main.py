#!/usr/bin/env python3
"""
Main CLI Entrypoint for Reinforcement Learning LunarLander Suite.
Provides a unified interface for training, testing, benchmarking, and visualizing
all 5 RL algorithms (Q-Learning, SARSA, Monte Carlo, CEM, and DQN).
"""

import argparse
import os
import sys
import subprocess

def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with return code {e.returncode}")
    except Exception as e:
        print(f"Error executing command: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Autonomous LunarLander Reinforcement Learning Suite"
    )
    
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "qlearning", "sarsa", "monte_carlo", "cem", "untrained"],
        default="dqn",
        help="Select RL algorithm (default: dqn)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train", "compare", "visualize"],
        default="test",
        help="Action mode: test (evaluate agent), train (train agent), compare (generate benchmark report), visualize (plot curves)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation/testing (default: 5)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Shortcut to run benchmark comparison across all algorithms"
    )
    
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.compare or args.mode == "compare":
        compare_script = os.path.join(root_dir, "compare_algorithms.py")
        print("Running benchmark evaluation across all 5 RL algorithms...")
        run_command([sys.executable, compare_script])
        return

    algo_dirs = {
        "dqn": "deep_q_learning",
        "qlearning": "qlearning",
        "sarsa": "sarsa",
        "monte_carlo": "monte_carlo",
        "cem": "cross_entropy"
    }
    
    if args.algo == "untrained":
        script_path = os.path.join(root_dir, "Untrained.py")
        print("Running Baseline Untrained Random Agent...")
        run_command([sys.executable, script_path])
        return

    target_dir = os.path.join(root_dir, algo_dirs[args.algo])
    
    if args.mode == "test":
        test_scripts = {
            "dqn": "testing_dqn.py",
            "qlearning": "testing_model.py",
            "sarsa": "testing_model.py",
            "monte_carlo": "testing_file.py",
            "cem": "testing_cem.py"
        }
        script_path = os.path.join(target_dir, test_scripts[args.algo])
        print(f"Testing {args.algo.upper()} agent ({script_path})...")
        run_command([sys.executable, script_path])
        
    elif args.mode == "train":
        train_scripts = {
            "dqn": "lunar_lander_dqn.py",
            "qlearning": "lunar_lander_qlearning.py",
            "sarsa": "lunar_lander_sarsa.py",
            "monte_carlo": "lunar_lander_monte_carlo.py",
            "cem": "cem_lunar_lander.py"
        }
        script_path = os.path.join(target_dir, train_scripts[args.algo])
        print(f"Training {args.algo.upper()} agent ({script_path})...")
        run_command([sys.executable, script_path])

    elif args.mode == "visualize":
        viz_scripts = {
            "dqn": "visualization.py",
            "qlearning": "visualization.py",
            "sarsa": "visualization.py",
            "monte_carlo": "visualization.py",
            "cem": "cem_lunar_lander.py"
        }
        script_path = os.path.join(target_dir, viz_scripts[args.algo])
        print(f"Visualizing learning metrics for {args.algo.upper()}...")
        run_command([sys.executable, script_path])

if __name__ == "__main__":
    main()
