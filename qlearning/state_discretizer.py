import numpy as np

class StateDiscretizer:
    def __init__(self, env):
        self.env = env
        
        # Define bin edges for each continuous observation dimension
        # The ranges are approximations based on the environment bounds
        # We need to balance precision with state-space size
        
        # Reduced bins to combat the Curse of Dimensionality
        # We need drastically fewer states for tabular Q-learning to converge in <5000 episodes
        
        self.bins = [
            # 1. Position X: [-1.5, 1.5]
            # Needs to distinguish: Far Left, Left, Center (Landing), Right, Far Right
            np.array([-0.5, -0.2, 0.2, 0.5]), 

            # 2. Position Y: [0, 1.5]
            # Needs: Ground, Very Low, Low, High
            np.array([0.1, 0.3, 0.6]),

            # 3. Velocity X: [-5, 5]
            # Needs: Moving Left Fast, Moving Left, Stable, Moving Right, Moving Right Fast
            np.array([-1.0, -0.2, 0.2, 1.0]),

            # 4. Velocity Y: [-5, 5]
            # Needs: Falling Fast, Falling, Hovering/Stable, Rising
            np.array([-1.0, -0.2, 0.2]),

            # 5. Angle: [-3.14, 3.14]
            # Needs: Tilted Left (Bad), Tilted Left (Slight), Upright, Tilted Right (Slight), Tilted Right (Bad)
            np.array([-0.2, -0.05, 0.05, 0.2]),

            # 6. Angular Velocity: [-5, 5]
            # Needs: Spinning Left, Stable, Spinning Right
            np.array([-0.1, 0.1]),

            np.array([0.5]),                    # Left leg
            np.array([0.5])                     # Right leg
        ]
        
        # Calculate state space size
        self.state_dims = [len(b) + 1 for b in self.bins]
        self.state_space_size = tuple(self.state_dims)
        
    def discretize(self, state):
        """
        Convert continuous state vector to a tuple of discrete indices.
        """
        # Unwrap state if it comes in a weird format (sometimes gym returns tuples)
        if isinstance(state, tuple):
            state = state[0]
            
        discrete_state = []
        for i, val in enumerate(state):
            # np.digitize returns index of the bin the value belongs to
            # bins[i] has N values, digitize returns 0..N
            # so we get N+1 possible indices
            idx = np.digitize(val, self.bins[i])
            discrete_state.append(idx)
            
        return tuple(discrete_state)
        
    def get_state_space_shape(self):
        return self.state_space_size
