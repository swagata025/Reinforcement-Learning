# Autonomous Lunar Lander using Tabular Q-Learning

## Introduction
We propose developing an autonomous control agent to safely land a spacecraft in the LunarLander-v2 environment. This project serves as a practical application of Reinforcement Learning (RL) fundamentals, specifically focusing on decision-making processes in unstable environments without relying on pre-built deep learning libraries.

## Objective
The primary goal is to understand and implement the core mechanics of RL—specifically state representation, reward structures, and policy optimization. By building the algorithm from scratch, we aim to gain a deeper insight into how an agent learns to balance immediate stability against long-term goals (landing).

## Approach: Tabular Q-Learning
We will solve the control problem using Tabular Q-Learning.

1. **From Scratch Implementation**: We implement the Q-learning algorithm manually to handle policy updates and exploration strategies ($\epsilon$-greedy).
2. **State Discretization**: Since the environment outputs continuous physics data (velocity, angle, position), we implement a "binning" strategy to convert these continuous values into discrete states that can be stored in a Q-table.
3. **Reward Maximization**: The agent is trained to maximize cumulative rewards by learning which thrust actions minimize fuel consumption and prevent crashes.
 
## Project Structure

- `lunar_lander_qlearning.py`: Main script for training and testing the agent using Tabular Q-learning.
- `state_discretizer.py`: Helper class to handle the conversion of continuous environmental observations into discrete bin indices.
- `visualization.py`: Generates learning curves from training logs.
- `requirements.txt`: Python dependencies.

## Installation

1. Install Python (3.8+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Windows users may need to install SWIG or Visual C++ Build Tools for Box2D support.*

## Usage

### Training
To train the agent from scratch:
```bash
python lunar_lander_qlearning.py
```
This will:
- Run for 2000 episodes (configurable).
- Save the Q-table to `q_table.pkl`.
- Save reward history to `rewards_history.npy`.

### Testing
To watch the trained agent:
1. Open `lunar_lander_qlearning.py`.
2. Comment out `train_agent()` and uncomment `test_agent()`.
3. Run the script again.

### Visualization
To see the learning progress:
```bash
python visualization.py
```
This generates `learning_curve.png`.

## Tools & Technologies
- **Python**: Core logic.
- **NumPy**: Q-table management and vector operations.
- **Gymnasium**: LunarLander-v2 simulation environment.
- **Matplotlib**: To visualize learning curves (Reward vs. Episode).

## Expected Outcome
A trained RL agent capable of consistently landing the spacecraft safely, accompanied by a performance analysis report showing the agent's learning progress.
