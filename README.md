# Autonomous LunarLander: Comparative Reinforcement Learning Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-brightgreen.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Solved](https://img.shields.io/badge/DQN_Status-Solved_(231.2_Avg)-success.svg)]()

A comprehensive benchmarking suite and custom implementation of **5 distinct Reinforcement Learning (RL) algorithms** to solve the unstable control problem of `LunarLander-v2 / v3`. 

This repository evaluates tabular methods, direct policy search, and deep function approximation—ranging from custom continuous-to-discrete state binning up to a lightweight Deep Q-Network (DQN) engineered using Scikit-Learn's `MLPRegressor` with Experience Replay and Target Network stabilization.

---

## Key Highlights & Architectural Features

- **5 Implemented RL Paradigms**:
  1. **Tabular Q-Learning**: Off-policy Temporal Difference (TD) learning with state space discretization.
  2. **SARSA**: On-policy Temporal Difference learning under $\epsilon$-greedy exploration.
  3. **Monte Carlo Control**: First-visit / Every-visit episodic learning without bootstrap bias.
  4. **Cross-Entropy Method (CEM)**: Black-box evolutionary policy search sampling over network weight space.
  5. **Deep Q-Network (DQN)**: Continuous state space function approximation via multi-layer perceptron regression (`MLPRegressor`), Experience Replay Buffer (200k samples), and target network stabilization.
- **Pure Scratch Implementation**: No heavy black-box RL libraries (`Stable-Baselines3`, `RLlib`). Core algorithms, experience replay, state binning, and target network updates are built directly using `NumPy` and standard Python data structures.
- **Empirical Benchmarks**: Standardized performance metrics across tens of thousands of episodes.
- **Unified CLI Tool (`main.py`)**: Seamless command-line interface for training, evaluation, visual testing, and benchmarking.

---

## Empirical Benchmark Results

Each algorithm was trained and evaluated on `LunarLander` under standard environment rewards ($+100$ for landing in goal zone, $-100$ for crashing, fuel consumption penalization). The environment is officially **SOLVED** when achieving a moving average reward of $\ge 200.0$ over 100 consecutive episodes.

| Algorithm | Method Type | Total Episodes / Iterations | Peak Reward | Final 100-Ep Moving Avg | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Q-Network (DQN)** | Deep Function Approximation | 2,000 | **318.4** | **231.2** | **SOLVED** |
| **Tabular Q-Learning** | Off-Policy TD ($\text{TD}(0)$) | 12,000 | 301.7 | 144.1 | Trained |
| **SARSA** | On-Policy TD ($\text{TD}(0)$) | 15,000 | 320.6 | 140.2 | Trained |
| **Monte Carlo Control** | Episodic MC Policy Iteration | 25,000 | 291.5 | -16.4 | High Variance |
| **Cross-Entropy Method** | Evolutionary Policy Search | 200 Generations | -79.0 | -106.6 | Linear Limit |
| **Untrained Agent** | Uniform Random Policy | N/A | -120.4 | -210.5 | Baseline |

### Learning Curves & Comparative Performance

![Algorithm Benchmark Comparison](plots/algorithm_comparison.png)

### Key Analytical Takeaways:
- **Sample Efficiency of Deep Function Approximation**: DQN solved the environment in **2,000 episodes**—6x fewer episodes than tabular methods—due to continuous state generalization.
- **Tabular Bottleneck (Discretization)**: Both Q-Learning and SARSA achieved strong control (peak scores >300, average ~140), but were constrained by state quantization limits.
- **On-Policy vs. Off-Policy Stability**: SARSA exhibited smoother, lower-variance policy updates near dangerous crash zones compared to off-policy Q-Learning due to taking current policy actions into account during updates.

---

## Theoretical & Algorithm Deep Dive

### 1. State Discretization (Binning Strategy)
`LunarLander-v2/v3` provides an 8-dimensional continuous state space:
$$S = [x, y, v_x, v_y, \theta, v_\theta, \text{leg}_1, \text{leg}_2]$$

For tabular methods (`qlearning`, `sarsa`, `monte_carlo`), continuous variables are quantized into discrete bin indices using a custom `StateDiscretizer` class:
$$\text{bin\_idx} = \text{digitize}(s_i, \text{bins}_i)$$
This reduces an infinite state space into a manageable, finite state index for lookup tables.

### 2. Tabular Q-Learning vs. SARSA Update Rules
- **Q-Learning (Off-Policy TD)**:
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$
- **SARSA (On-Policy TD)**:
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

### 3. Deep Q-Network (DQN) Architecture
- **State Representation**: 8-dimensional continuous vector (no quantization needed).
- **Network**: Multi-Layer Perceptron (`MLPRegressor` with twin hidden layers $[256, 256]$ and ReLU activations).
- **Replay Buffer**: Capacity $N = 200,000$ transitions to break temporal correlation:
  $$\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, \text{done}_t)\}$$
- **Target Network Synchronization**: Soft target updates executed every 15 episodes to stabilize Bellman loss target calculations:
  $$y_j = r_j + \gamma \max_{a'} Q_{\text{target}}(s'_j, a')$$

---

## Repository Structure

```text
Reinforcement-Learning/
├── main.py                     # Unified CLI tool (Train, Test, Benchmark)
├── compare_algorithms.py       # Benchmark evaluation & plot generator
├── hyperparameter_tracker.py   # Hyperparameter sensitivity tracker & logger
├── app.py                      # Interactive Streamlit web app
├── Untrained.py                # Baseline random agent runner
├── Dockerfile                  # Container build instructions
├── requirements.txt            # Project dependencies
├── plots/
│   ├── algorithm_comparison.png# Benchmark visual comparison chart
│   └── hyperparameter_sensitivity.png # Hyperparameter sensitivity plot
├── experiments/                # Generated hyperparameter CSV/JSON logs
├── deep_q_learning/            # Deep Q-Network (DQN) implementation
├── qlearning/                  # Tabular Q-Learning implementation
├── sarsa/                      # SARSA implementation
├── monte_carlo/                # Monte Carlo Control implementation
├── cross_entropy/              # Cross-Entropy Method (CEM) implementation
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions automated CI verification
```

---

## Installation & Getting Started

### 1. Prerequisites & Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/swagata025/Reinforcement-Learning.git
cd Reinforcement-Learning
pip install -r requirements.txt
```

### 2. Using the Unified CLI (`main.py`)

#### Run Benchmark Comparison Report
```bash
python main.py --compare
```
*Generates the ASCII benchmark table in the terminal and updates `plots/algorithm_comparison.png`.*

#### Test / Visualize a Trained Agent
To watch the pre-trained **DQN** agent land in a live rendering window:
```bash
python main.py --algo dqn --mode test
```
To test **Tabular Q-Learning** or **SARSA**:
```bash
python main.py --algo qlearning --mode test
python main.py --algo sarsa --mode test
```

#### Train an Agent from Scratch
To train the **DQN** agent:
```bash
python main.py --algo dqn --mode train
```

---

## Interactive Web Demo (Streamlit)
Launch the interactive web dashboard to test agents, adjust environment physics (gravity, wind, turbulence), and compare model metrics in your browser:
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`.*

---

## Hyperparameter Experiment Tracker
Run standard hyperparameter sensitivity grid trials across learning rates ($\alpha$), discount factors ($\gamma$), and buffer sizes ($N$):
```bash
python hyperparameter_tracker.py
```
*Generates `experiments/hyperparameter_experiments.csv`, `experiments/hyperparameter_summary.json`, and `plots/hyperparameter_sensitivity.png`.*

---

## Docker Deployment
Containerize and execute the full environment without installing local dependencies or SWIG build tools:

```bash
# Build Docker image
docker build -t lunar-lander-rl .

# Run benchmark report in container
docker run --rm lunar-lander-rl
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
