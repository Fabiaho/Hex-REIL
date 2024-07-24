import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from alphaZero.Arena import Arena
from alphaZero.MCTS import MCTS
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.create_competitor_nnet()  # Create the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()
        self.current_temp = self.args.initial_temp  # Initialize temperature
        self.losses = []  # List to store training losses
        self.win_rates = []  # List to store win rates
        
    def create_competitor_nnet(self):
        # Create a competitor network with the same dimensions as the original network
        return self.nnet.__class__(self.nnet.fc1.in_features, self.nnet.fc1.out_features, self.nnet.fc3.out_features)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.get_init_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            temp = self.current_temp if episodeStep < self.args.temp_threshold else 0

            pi = self.mcts.get_action_prob(canonicalBoard, temp=temp)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            valid_moves = self.game.get_valid_moves(board, self.curPlayer)
            if valid_moves[action] == 0:
                continue

            board, self.curPlayer = self.game.get_next_state(board, self.curPlayer, action)

            r = self.game.get_game_ended(board, self.curPlayer)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            
            if self.args.visualize_self_play:
                self.game.display(board)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.num_iters + 1):
            print(100*'-')
            print(f"Training iteration {i}/{self.args.num_iters}")
            log.info(f'Starting Iter #{i} ...')

            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlen_of_queue)

                for _ in tqdm(range(self.args.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset mcts search tree
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.num_iters_for_train_examples_history:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            trainExamples = [e for history in self.trainExamplesHistory for e in history]
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Record training losses
            losses = self.nnet.train_model(trainExamples)  # Use the correct training method
            self.losses.append(losses)

            pmcts = MCTS(self.game, self.pnet, self.args)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)
            pwins, nwins = arena.play_games(self.args.arena_compare)

            win_rate = float(nwins) / (pwins + nwins)
            self.win_rates.append(win_rate)

            log.info(f'NEW/PREV WINS : {nwins} / {pwins}')
            if pwins + nwins == 0 or win_rate < self.args.update_threshold:
                print('REJECTING the new model')
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING the new model')
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            # Decay the temperature
            self.current_temp = max(self.args.final_temp, self.current_temp * self.args.temp_decay_factor)
            
            
    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def plot_training_progress(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for epoch_losses in self.losses:
            plt.plot(epoch_losses)
        plt.title('Training Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.win_rates)
        plt.title('Win Rates')
        plt.xlabel('Iterations')
        plt.ylabel('Win Rate')

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, iteration):
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))

    def load_checkpoint(self, iteration):
        self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))

    def _pit(self, pnet, nnet):
        pmcts = MCTS(self.game, pnet, self.args)
        nmcts = MCTS(self.game, nnet, self.args)

        arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                      lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)

        return arena.play_games(self.args.arena_compare)