from abc import ABC
from typing import Callable
from pathlib import Path
import torch
import numpy as np

from deepq_hex_convolutional import ConvolutionalPolicy
from deepq_hex_convpos import ConvolutionalPositionalPolicy
from deepq_hex_feedforward import FeedForwardPolicy
from deepq_hex_transformer import TransformerPolicy
from deepq_hex_transformer2 import TransformerPolicy as TransformerPolicy2

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
        self.model = model.to("cuda")
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
        state = state.to("cuda")
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
    

class RandomAgent(BaseAgent):
    def __init__(self, size: int = 7, player: int = 1):
        super(RandomAgent, self).__init__(torch.nn.Module(), size, player)
        
    def __call__(self, board: list[list[int]], moves: list[int]) -> int:
        return moves[np.random.randint(len(moves))]


class ConvAgent(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = ConvolutionalPolicy(size*size, 128, layers=3)
        super(ConvAgent, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)
            
            
class ConvPosAgent(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = ConvolutionalPositionalPolicy(size*size, 128, layers=3)
        super(ConvPosAgent, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)
        
        
class FeedForwardAgent(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = FeedForwardPolicy(size*size, 128, layers=2)
        super(FeedForwardAgent, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)
        
        
class TransformerAgent(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = TransformerPolicy(size*size, 128, 128//16, 4)
        super(TransformerAgent, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)
            
            
class TransformerAgent2(BaseAgent):
    def __init__(self, parameters_file: Path|str|None, size: int = 7, player: int = 1):
        model = TransformerPolicy2(size*size, 128, 128//16, 4)
        super(TransformerAgent2, self).__init__(model, size, player)
        if parameters_file is not None:
            self.load_parameters(parameters_file)
        
        
    