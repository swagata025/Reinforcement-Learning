import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="LunarLander RL Control Suite",
    page_icon="🚀",
    layout="wide"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_benchmark_data():
    paths = {
        'Deep Q-Network (DQN)': os.path.join(SCRIPT_DIR, 'deep_q_learning', 'data', 'rewards_history.npy'),
        'Tabular Q-Learning': os.path.join(SCRIPT_DIR, 'qlearning', 'data', 'rewards_history.npy'),
        'SARSA (On-Policy TD)': os.path.join(SCRIPT_DIR, 'sarsa', 'data', 'rewards_history.npy'),
        'Monte Carlo Control': os.path.join(SCRIPT_DIR, 'monte_carlo', 'data', 'mc_rewards_history.npy'),
        'Cross-Entropy Method': os.path.join(SCRIPT_DIR, 'cross_entropy', 'data', 'rewards_history.npy')
    }
    data = {}
    for name, path in paths.items():
        if os.path.exists(path):
            data[name] = np.load(path)
    return data

@st.cache_data
def load_hyperparams():
    csv_path = os.path.join(SCRIPT_DIR, "experiments", "hyperparameter_experiments.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def main():
    st.title("🚀 Autonomous LunarLander: Reinforcement Learning Benchmark Suite")
    st.markdown("An interactive web interface for evaluating 5 distinct RL paradigms trained on Gymnasium's `LunarLander-v3` environment.")
    
    benchmark_data = load_benchmark_data()
    hyperparams_df = load_hyperparams()
    
    # -----------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -----------------------------------------------------------------------------
    st.sidebar.header("🕹️ Agent Configuration")
    
    algo_option = st.sidebar.selectbox(
        "Select RL Model",
        [
            'Deep Q-Network (DQN)',
            'Tabular Q-Learning',
            'SARSA (On-Policy TD)',
            'Monte Carlo Control',
            'Cross-Entropy Method',
            'Untrained Random Baseline'
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Physics Controls")
    gravity = st.sidebar.slider("Gravity (m/s²)", -12.0, -2.0, -10.0, 0.5)
    enable_wind = st.sidebar.checkbox("Enable Environment Wind", value=False)
    enable_turbulence = st.sidebar.checkbox("Enable Wind Turbulence", value=False, disabled=not enable_wind)

    # -----------------------------------------------------------------------------
    # METRICS DISPLAY
    # -----------------------------------------------------------------------------
    st.subheader(f"📊 Model Overview: {algo_option}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if algo_option in benchmark_data:
        rewards = benchmark_data[algo_option]
        final_avg = np.mean(rewards[-100:])
        peak = np.max(rewards)
        episodes = len(rewards)
        status = "🟢 SOLVED (>= 200)" if final_avg >= 200 else ("🟡 TRAINED" if final_avg > 100 else "🔴 UNCONVERGED")
    elif algo_option == 'Untrained Random Baseline':
        final_avg, peak, episodes, status = -210.5, -120.4, "N/A", "🔴 BASELINE"
    else:
        final_avg, peak, episodes, status = 0.0, 0.0, 0, "UNKNOWN"

    col1.metric("Status", status)
    col2.metric("Final 100-Ep Moving Avg", f"{final_avg:.1f}")
    col3.metric("Peak Reward Achieved", f"{peak:.1f}")
    col4.metric("Total Training Episodes", f"{episodes:,}")

    # -----------------------------------------------------------------------------
    # TABS FOR BENCHMARKS AND SIMULATION
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📈 Comparative Benchmark Curves", "🧪 Hyperparameter Sensitivity", "🧠 Model Architecture & Math"])
    
    with tab1:
        st.write("### Rolling 100-Episode Moving Average Reward Curves")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        colors = {
            'Tabular Q-Learning': '#1f77b4',
            'SARSA (On-Policy TD)': '#ff7f0e',
            'Monte Carlo Control': '#2ca02c',
            'Cross-Entropy Method': '#d62728',
            'Deep Q-Network (DQN)': '#9467bd'
        }
        
        for name, rewards in benchmark_data.items():
            win = 100
            m_avg = np.convolve(rewards, np.ones(win)/win, mode='valid')
            alpha = 1.0 if name == algo_option else 0.3
            lw = 2.5 if name == algo_option else 1.2
            ax.plot(m_avg, label=name, color=colors.get(name, '#333333'), alpha=alpha, linewidth=lw)
            
        ax.axhline(y=200.0, color='red', linestyle='--', label='Environment Solved (200+)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(loc='lower right')
        st.pyplot(fig)

    with tab2:
        st.write("### Experimental Hyperparameter Sweep Results")
        if hyperparams_df is not None:
            st.dataframe(hyperparams_df, use_container_width=True)
            sens_plot_path = os.path.join(SCRIPT_DIR, 'plots', 'hyperparameter_sensitivity.png')
            if os.path.exists(sens_plot_path):
                st.image(sens_plot_path, caption="DQN Learning Rate and Buffer Size Sensitivity Analysis")
        else:
            st.info("Run `python hyperparameter_tracker.py` to generate hyperparameter logs.")

    with tab3:
        if algo_option == 'Deep Q-Network (DQN)':
            st.markdown(r"""
            **Deep Q-Network (DQN)**
            - **Function Approximator**: Scikit-Learn `MLPRegressor` with hidden layers $(256, 256)$ and ReLU activations.
            - **State Representation**: 8D continuous state vector.
            - **Stabilization Mechanisms**: Experience Replay Buffer ($N=200,000$) + Target Network synchronization every 15 episodes.
            - **Loss Target**: $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$
            """)
        elif 'Q-Learning' in algo_option:
            st.markdown(r"""
            **Tabular Q-Learning (Off-Policy TD)**
            - **State Representation**: Custom `StateDiscretizer` digitizing continuous variables into bin indices.
            - **Update Equation**: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
            """)
        elif 'SARSA' in algo_option:
            st.markdown(r"""
            **SARSA (On-Policy TD)**
            - **State Representation**: Continuous state discretization via binning.
            - **Update Equation**: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$ where $a'$ is chosen via $\epsilon$-greedy policy.
            """)
        else:
            st.markdown(f"**Model Description**: Implemented within the `{algo_option}` module.")

    st.markdown("---")
    st.caption("Reinforcement Learning LunarLander Control Suite | Built with Gymnasium & Streamlit")

if __name__ == "__main__":
    main()
