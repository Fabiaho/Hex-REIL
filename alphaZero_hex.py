from alphaZero.Net import NeuralNet
from alphaZero.HexWrapper import HexGameWrapper
from alphaZero.Coach import Coach
from alphaZero.Arena import Arena
from alphaZero.MCTS import MCTS
import numpy as np

class Args:
    def __init__(self):
        self.num_iters = 1000 #specifies the number of iterations of self-play and training.
        self.num_eps = 100 # number of self-play episodes per iteration
        self.maxlen_of_queue = 200000 # maximum length of the deque that stores training examples from self-play episodes.
        self.num_mcts_sims = 25 # number of MCTS simulations to run for each move to explore the game tree and determine the best move.
        self.cpuct = 1
        self.num_iters_for_train_examples_history = 20 # specifies how many iterations of training examples should be kept in the training history.
        self.visualize_self_play = False # control visualization of self-play
        
        # neural network exchange
        self.arena_compare = 40 # number of games to play in the arena when comparing the new neural network against the old neural network one.
        self.update_threshold = 0.6 # decide whether the new neural network (after training) should replace the old one | 0.6 means that the new network must win at least 60% of the games
        self.checkpoint = './checkpoints'
        
        #exploration and exploitation
        self.temp_threshold = 15 # parameter controls the exploration-exploitation trade-off. higher temperature is used to encourage exploration
        self.initial_temp = 1.0  # Initial temperature for exploration
        self.final_temp = 0.1  # Final temperature for exploitation
        self.temp_decay_factor = 0.95  # Decay factor for temperature
        
        
def random_player(board, game):
    valid_moves = game.get_valid_moves(board, 1)
    valid_moves_indices = [i for i, valid in enumerate(valid_moves) if valid == 1]
    return np.random.choice(valid_moves_indices)

if __name__ == "__main__":
    args = Args()
    args.visualize_self_play = False  # Enable self-play visualization
    board_size = 5  # You can set this to 5, 7, or any other valid size
    game = HexGameWrapper(size=board_size)
    nnet = NeuralNet(input_dim=board_size * board_size, hidden_dim=128, output_dim=board_size * board_size)
    coach = Coach(game, nnet, args)
    
    ###### learn
    coach.learn()
    # Plot the training progress
    coach.plot_training_progress()

    # Load the trained neural network
    nnet = NeuralNet(input_dim=board_size * board_size, hidden_dim=128, output_dim=board_size * board_size)
    nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    
    # Define the MCTS player
    mcts_player = lambda board: np.argmax(MCTS(game, nnet, args).get_action_prob(board, temp=0))
    
    # Define the random player
    random_player_fn = lambda board: random_player(board, game)
    
    # Create an arena to play games between the MCTS player and the random player
    arena = Arena(mcts_player, random_player_fn, game)
    num_games = 10  # Number of games to play
    mcts_wins, random_wins = arena.play_games(num_games, verbose=True)
    
    # Print the results
    print(f"Results after {num_games} games:")
    print(f"MCTS Player Wins: {mcts_wins}")
    print(f"Random Player Wins: {random_wins}")