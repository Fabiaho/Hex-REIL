
from abc import ABC
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


class ConvolutionalPositionalPolicy(nn.Module):
    def __init__(self, size: int, width: int = 32, n_heads: int = 4, layers: int = 2):
        super(ConvolutionalPositionalPolicy, self).__init__()
        self.input_layer = nn.Conv2d(4, width, kernel_size=3, padding=1)
        side = int(math.sqrt(size))
        pos_range = torch.linspace(-1, 1, side).unsqueeze(0)
        pos_x = pos_range.repeat(1, side)
        pos_y = pos_range.t().repeat(1, side).reshape(1, side*side)
        pos_z = -pos_x - pos_y
        self.pos_encoding = torch.stack([pos_x, pos_y, pos_z], dim=1)
        self.hidden_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=5, padding=2) for _ in range(layers)]
        )
        self.output_layer = nn.Conv2d(width, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        side = int(np.sqrt(x.shape[1]))
        valid_actions = (x.view(-1, 1, side, side).round().int() == 0).float()
        pos_encoding = self.pos_encoding.repeat(x.size(0), 1, 1).to(x.device)
        x = torch.cat([x.unsqueeze(1), pos_encoding], dim=1)
        x = x.view(-1, 4, side, side)
        x = self.input_layer(x)
        shortcut = x
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            x = x + shortcut
        x = self.output_layer(x)
        x = nn.Sigmoid()(x) * valid_actions    
        return x.view(-1, side * side)
    

class BaseAgent(Callable[[list[list[int]], list[int]], int], ABC):
    """
    Base class for agents that play Hex. The agent is callable and takes a board and a list of possible moves as input.
    Thus, it can be directly used in the `hexPosition` class. This is an abstract base class that should be extended
    by concrete agents using a specific model architecture.
    
    Parameters
    ----------
    model: torch.nn.Module
        The model that predicts the next move.
    size: int
        The size of the board. Default is 7.
    player: int
        The player that the agent represents, should be either 1 or -1. Default is 1.
    """
    
    def __init__(self, model: torch.nn.Module, size: int = 7, player: int = 1):
        # self.model = model.to("cuda")
        self.model = model
        self.size = size
        self.player = player
        self.path: Path|str|None = None
    
    @property
    def recode(self) -> bool:
        return self.player == -1
        
    def load_parameters(self, parameters_file: Path|str):
        self.model.load_state_dict(torch.load(parameters_file))
        self.path = parameters_file
    
    def __call__(self, board: list[list[int]], moves: list[int]) -> int:
        if self.recode:
            board = self._recode_black_as_white(board)
        state = torch.tensor(np.array(board).flatten()).float().unsqueeze(0)
        # state = state.to("cuda")
        with torch.no_grad():
            action = self.model(state).argmax().item()
        action = (action // self.size, action % self.size)
        if self.recode:
            action = self._recode_coordinates(action)
        if action not in moves:
            action = moves[0]
        return action
    
    def _recode_black_as_white(self, board: list[list[int]], print=False, invert_colors=True) -> list[list[int]]:
        """
        Returns a board where black is recoded as white and wants to connect horizontally.
        This corresponds to flipping the board along the south-west to north-east diagonal and swapping colors.
        This may be used to train AI players in a 'color-blind' way.
        """
        flipped_board = [[0 for i in range(self.size)] for j in range(self.size)]
        # flipping and color change
        for i in range(self.size):
            for j in range(self.size):
                if board[self.size - 1 - j][self.size - 1 - i] == 1:
                    flipped_board[i][j] = -1
                if board[self.size - 1 - j][self.size - 1 - i] == -1:
                    flipped_board[i][j] = 1
        return flipped_board

    def _recode_coordinates(self, coordinates):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recode_black_as_white.
        """
        assert (
            0 <= coordinates[0] and self.size - 1 >= coordinates[0]
        ), "There is something wrong with the first coordinate."
        assert (
            0 <= coordinates[1] and self.size - 1 >= coordinates[1]
        ), "There is something wrong with the second coordinate."
        return (self.size - 1 - coordinates[1], self.size - 1 - coordinates[0])
    
    
class ConvPosAgent(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = ConvolutionalPositionalPolicy(size*size, 128, layers=3)
        super(ConvPosAgent, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)



best_model_path = Path(__file__).parent / "best_agent.pth"
actor = ConvPosAgent(best_model_path, size=7)

def agent(board: list[list[int]], allowed_moves: list[tuple[int, int]]) -> tuple[int, int]:
    sum_stones = sum(sum(row) for row in board)
    if sum_stones > 0:
        actor.player = -1
    else:
        actor.player = 1
    pass
    return actor(board, allowed_moves)