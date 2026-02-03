import gymnasium as gym
from discretize import *
from qLearningAgent import *

env = gym.make("LunarLander-v3")
discretizer = StateDiscretizer(env)
agent = QLearningAgent(discretizer, env.action_space.n)


save_count = 100
for episode in range(10000):
    if (episode%save_count == 0):
        agent.save('q-table.pkl')

    raw_state, _ = env.reset()
    state = discretizer.discretize(raw_state)
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_raw_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = discretizer.discretize(next_raw_state)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        if done: break