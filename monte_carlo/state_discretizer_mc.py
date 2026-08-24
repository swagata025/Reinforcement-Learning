import numpy as np

class StateDiscretizerMC:
    def __init__(self, env):
        self.env = env
        
        # Refined Binning Strategy for Monte Carlo
        # Aim: Reduce total state space while maintaining precision where it matters (Lander Center & Stability)
        
        self.bins = [
            # 1. Position X: [-1.5, 1.5]
            # Focus on precise centering. Target is 0.0
            # Bins: Far Left, Left Approach, On Target (Left), On Target (Right), Right Approach, Far Right
            np.array([-0.5, -0.1, 0.0, 0.1, 0.5]), 

            # 2. Position Y: [0, 1.5]
            # Focus on proximity to ground.
            # Bins: Ground/Landed (<0.1), Final Approach (0.1-0.3), Descent (0.3-0.6), High
            np.array([0.1, 0.3, 0.6]),

            # 3. Velocity X: [-5, 5]
            # Needs to be stable near 0.
            # Bins: Moving Left, Stable (-0.15 to 0.15), Moving Right
            np.array([-0.2, 0.2]),

            # 4. Velocity Y: [-5, 5]
            # Key for survival.
            # Bins: Crash (< -0.8), Apply Engines (< -0.4), Safe Descent (< -0.05), Hover/Rise (>= -0.05)
            # Note: -0.05 is chosen to capture very slow descent/hovering.
            np.array([-1.0, -0.5, 0.0]),

            # 5. Angle: [-3.14, 3.14]
            # Stability is paramount.
            # Bins: Danger Left, Tilt Left, Stable, Tilt Right, Danger Right
            np.array([-0.2, -0.05, 0.05, 0.2]),

            # 6. Angular Velocity: [-5, 5]
            # Just separate spinning from stable.
            np.array([-0.15, 0.15]),

            # 7. Left leg contact
            np.array([0.5]),
            # 8. Right leg contact
            np.array([0.5]) 
        ]
        
        self.state_dims = [len(b) + 1 for b in self.bins]
        self.state_space_size = tuple(self.state_dims)
        
    def discretize(self, state):
        if isinstance(state, tuple):
            state = state[0]
            
        discrete_state = []
        for i, val in enumerate(state):
            idx = np.digitize(val, self.bins[i])
            discrete_state.append(idx)
            
        return tuple(discrete_state)
        
    def get_state_space_shape(self):
        return self.state_space_size
