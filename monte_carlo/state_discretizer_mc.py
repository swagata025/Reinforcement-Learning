import numpy as np

class StateDiscretizerMC:
    def __init__(self, env):
        self.env = env
        
        # "Smart Coarse" binning strategy
        # Reduces state space size to ~13k states
        # Emphasizes Velocity and Angle over Position for better generalization
        self.bins = [
            # 1. Position X: [-1.5, 1.5] - Coarse (Left, Center, Right)
            np.array([-0.3, 0.3]), 

            # 2. Position Y: [0, 1.5] - Coarse (Low, Mid, High)
            np.array([0.3, 0.6]),

            # 3. Velocity X: [-5, 5] - Important for stability
            np.array([-0.4, -0.1, 0.1, 0.4]),

            # 4. Velocity Y: [-5, 5] - Critical for landing
            # Breaks down into: Fast Descent, Controlled Descent, Hover, Rise
            np.array([-0.9, -0.5, -0.2, 0.2]),

            # 5. Angle: [-3.14, 3.14] - Critical for staying upright
            np.array([-0.2, -0.05, 0.05, 0.2]),

            # 6. Angular Velocity: [-5, 5] - Coarse
            np.array([-0.2, 0.2]),

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
