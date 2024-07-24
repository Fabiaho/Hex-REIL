from hex_engine import hexPosition
from copy import deepcopy

class HexGameWrapper:
    def __init__(self, size):
        self.size = size

    def get_init_board(self):
        return hexPosition(size=self.size).board

    def get_canonical_form(self, board, player):
        return board if player == 1 else [[-cell for cell in row] for row in board]

    def get_next_state(self, board, player, action):
        hex_game = hexPosition(size=self.size)
        hex_game.board = deepcopy(board)
        hex_game.player = player

        # Check if the game is already won
        if hex_game.winner != 0:
            raise ValueError("The game has already been won.")

        x, y = action // self.size, action % self.size
        if hex_game.board[x][y] != 0:
            raise ValueError("Invalid move: cell is already occupied.")
        
        hex_game.moove((x, y))
        
        # Evaluate the board after the move to check if there's a winner
        hex_game.evaluate()
        #if hex_game.winner != 0:
        #    self.display(hex_game.board)
        
        return hex_game.board, -player

    def get_valid_moves(self, board, player):
        hex_game = hexPosition(size=self.size)
        hex_game.board = deepcopy(board)
        valid_moves = hex_game.get_action_space()
        moves = [0] * (self.size * self.size)
        for move in valid_moves:
            moves[move[0] * self.size + move[1]] = 1
        return moves

    def get_game_ended(self, board, player):
        hex_game = hexPosition(size=self.size)
        hex_game.board = deepcopy(board)
        hex_game.evaluate()
        if hex_game.winner == 1:
            #print("White wins!")
            #self.display(hex_game.board)
            return 1
        elif hex_game.winner == -1:
            #print("Black wins!")
            #self.display(hex_game.board)
            return -1
        else:
            return 0

    def get_action_size(self):
        return self.size * self.size

    def string_representation(self, board):
        return str(board)

    def display(self, board):
        hex_game = hexPosition(size=self.size)
        hex_game.board = deepcopy(board)
        hex_game.print()

    def get_symmetries(self, board, pi):
        return [(board, pi)]
