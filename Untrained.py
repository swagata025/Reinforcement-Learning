import gymnasium as gym

def test_untrained_model(episodes=5):
    # Create environment with human rendering enabled
    env = gym.make("LunarLander-v3", render_mode="human")
    
    print(f"Visualizing {episodes} episodes with a random agent...")
    print("Close the window to stop early.")

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (terminated or truncated):
            # Take a random action
            action = env.action_space.sample()
            
            # Step the environment
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Steps: {step}")

    env.close()

if __name__ == "__main__":
    test_untrained_model()
