import numpy as np

class Arena:
    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self, verbose=False):
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.get_init_board()
        iteration = 0

        while self.game.get_game_ended(board, cur_player) == 0:
            iteration += 1
            
            if verbose:
                print("Turn ", str(iteration), "Player ", str(cur_player))
                self.game.display(board)

            action = players[cur_player + 1](board)
            board, cur_player = self.game.get_next_state(board, cur_player, action)

        if verbose:
            print("Game over: Turn ", str(iteration), "Result ", str(self.game.get_game_ended(board, 1)))
            self.game.display(board)

        return self.game.get_game_ended(board, 1)

    def play_games(self, num, verbose=False):
        one_won = 0
        two_won = 0

        for _ in range(num):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1

        return one_won, two_won
    
    def random_player(self, board):
        valid_moves = self.game.get_valid_moves(board, 1)
        valid_moves_indices = [i for i, valid in enumerate(valid_moves) if valid == 1]
        return np.random.choice(valid_moves_indices)
