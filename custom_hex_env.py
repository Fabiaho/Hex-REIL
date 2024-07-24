import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hex_engine import hexPosition
from HexViewer import HexViewer

class CustomHexEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, size=5, opponent_type="random", opponent_model=None):
        super(CustomHexEnv, self).__init__()
        self.hex_engine = hexPosition(size=size)

        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.opponent_model = opponent_model

        # Initialize HexViewer only if the render mode is 'human'
        self.viewer = (
            HexViewer(800, 600, self.hex_engine.size, self.hex_engine.size)
            if render_mode == "human"
            else None
        )

        # Define action and observation space
        self.action_space = spaces.Discrete(self.hex_engine.size**2)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.hex_engine.size, self.hex_engine.size),
            dtype=np.int32,
        )
        self.reset()

    def step(self, action):
        row, col = divmod(action, self.hex_engine.size)
        invalid_move = False

        # Check if the action is valid (the cell is empty)
        if self.hex_engine.board[row][col] != 0:
            invalid_move = True

        if not invalid_move:
            # Perform the move
            self.hex_engine.moove((row, col))

        if self.hex_engine.winner == 0:
            # Opponent's move if not self-play
            if self.opponent_type == "self":
                opponent_action = self._get_self_play_action()
            else:
                opponent_action = self._get_opponent_action()

            opponent_coordinates = self.hex_engine.scalar_to_coordinates(opponent_action)
            assert (
                self.hex_engine.board[opponent_coordinates[0]][opponent_coordinates[1]] == 0
            ), f"Invalid move by opponent, cell occupied: {opponent_coordinates}"
            self.hex_engine.moove(opponent_coordinates)

        done = self.hex_engine.winner != 0

        reward = (
            100
            if self.hex_engine.winner == 1
            else -100 if self.hex_engine.winner == -1 else -1
        )
        
        obs = np.array(self.hex_engine.board, dtype=np.int32)
        info = {}
        
        if invalid_move:
            reward -= 10
            info = {"error": "Invalid action"}

        return obs, reward, done, False, info

    def _get_opponent_action(self):
        if self.opponent_type == "random":
            return np.random.choice(
                [
                    self.hex_engine.coordinate_to_scalar(move) for move in self.hex_engine.get_action_space()
                ]
            )
            
        elif self.opponent_type == "trained" and self.opponent_model:
            
            valid_action_indices = [self.hex_engine.coordinate_to_scalar(action) for action in  self.hex_engine.get_action_space()]
           
            observation = np.array(self.hex_engine.board, dtype=np.int32)
            
            action, _ = self.opponent_model.predict(observation, deterministic=True)
            #print(action)
            if action not in valid_action_indices:
                action = np.random.choice(valid_action_indices)
            
            return action
        else:
            raise ValueError("Invalid opponent type or opponent model not provided")

    def _get_self_play_action(self):
        scalar_actions = [
            self.hex_engine.coordinate_to_scalar(move) for move in self.hex_engine.get_action_space()
        ]
        return np.random.choice(scalar_actions)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.hex_engine.reset()
        return np.array(self.hex_engine.board, dtype=np.int32), {}

    def render(self, mode="human"):
        if mode == "human" and self.viewer:
            self.viewer.render(self.hex_engine.board)
        elif mode == "rgb_array":
            return self._get_rgb_array()
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _get_rgb_array(self):
        # Convert the board to an RGB array (example: black = 0, white = 255, empty = 128)
        color_map = {0: [128, 128, 128], 1: [255, 255, 255], -1: [0, 0, 0]}
        return np.array(
            [[color_map[cell] for cell in row] for row in self.hex_engine.board],
            dtype=np.uint8,
        )

    def valid_action_mask(self):
        mask = np.zeros(self.hex_engine.size**2, dtype=np.int32)
        for move in self.hex_engine.get_action_space():
            mask[self.hex_engine.coordinate_to_scalar(move)] = 1
        return mask
    
# Register the environment
from gymnasium import register

register(
    id="CustomHex-v0",
    entry_point="custom_hex_env:CustomHexEnv",
)
