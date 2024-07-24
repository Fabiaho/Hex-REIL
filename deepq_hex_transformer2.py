# deepq_hex.py

from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict
from itertools import count
import matplotlib.pyplot as plt
from hex_engine import hexPosition
import math
import os
import pandas as pd

# Convenience class to keep transition data straight
# is used inside 'replayMemory'
transitionData = namedtuple("Transition", ["state", "action", "next_state", "reward"])


class HexAgent:
    def __init__(self, model_path, size=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.n_actions = size * size
        self.n_observations = size * size

        # Initialize the network
        self.policy_net = make_torch_net(
            input_length=self.n_observations,
            width=128,  # assuming the same width and hidden layers as in training
            output_length=self.n_actions,
            hidden=4,
        ).to(self.device)

        # Load the trained model
        self.policy_net.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.policy_net.eval()  # Set the network to evaluation mode

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            action_values = self.policy_net(state)
        action = action_values.argmax().view(1, 1)
        return action.item()

    def play(self, env: hexPosition):
        state = np.array(env.recode_black_as_white()).flatten()
        action = self.get_action(state)
        coordinate = (action // self.size, action % self.size)
        recoded_action = env.recode_coordinates(coordinate)
        env.moove(recoded_action)


class replayMemory(object):
    """
    Store transitions consisting of 'state', 'action', 'next_state', 'reward'.
    """

    def __init__(self, length: int):
        self.memory = deque([], maxlen=length)

    def save(self, state, action, next_state, reward):
        self.memory.append(transitionData(state, action, next_state, reward))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TransformerPolicy(nn.Module):
    def __init__(self, size: int, width: int = 32, n_heads: int = 4, layers: int = 2):
        super(TransformerPolicy, self).__init__()
        self.pos_encoding = nn.Embedding(size, width)
        self.input_embedding = nn.Embedding(3, width)
        self.norm = nn.LayerNorm(width)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=width, nhead=n_heads, dim_feedforward=width*4, dropout=0.0, batch_first=True),
            num_layers=layers
        )
        self.output_layer = nn.Linear(width, 1)
        # self.eps = torch.tensor(1e-6, dtype=torch.float32)
        
    def forward(self, x):
        x = x.round().int()
        valid_actions = (x == 0).float()
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_encoding = self.pos_encoding(pos)
        x = self.input_embedding(x+1) + pos_encoding
        x = self.transformer(x)
        x = nn.ReLU()(x)
        x = self.output_layer(x).squeeze(2)
        x = nn.Sigmoid()(x) * valid_actions
        # x = x / torch.maximum(x.sum(1, keepdim=True), self.eps)
        return x


def make_torch_net(input_length: int, width: int, output_length: int, hidden=1):
    net = TransformerPolicy(size=input_length, width=width, n_heads=width//16, layers=hidden)
    print(net)
    return net


class deepQ(object):
    """
    Deep Q Learning wrapper.
    """

    def __init__(self, size=7, memory_length=1000):
        self.env = hexPosition(size=size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = replayMemory(length=memory_length)
        self.size = size
        self.n_actions = size * size
        self.n_observations = size * size
        self.episode_durations = []
        self.eps = torch.tensor(1e-6, dtype=torch.float32)
    

    def get_state(self):
        return np.array(self.env.board).flatten()

    def initialize_networks(self, width=128, hidden=4):
        self.policy_net = make_torch_net(
            input_length=self.n_observations,
            width=width,
            output_length=self.n_actions,
            hidden=hidden,
        ).to(self.device)
        self.target_net = make_torch_net(
            input_length=self.n_observations,
            width=width,
            output_length=self.n_actions,
            hidden=hidden,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _eps_greedy_action(self, state, eps_threshold):
        if random.random() > eps_threshold:
            # Select the action with the highest value among valid actions
            with torch.no_grad():
                action_values = self.policy_net(state)
            action = action_values.argmax().view(1, 1)
        else:
            # Select a random valid action
            valid_actions_mask = torch.tensor(
                np.asarray(self.env.board).flatten() == 0, device=self.device
            )
            valid_actions = valid_actions_mask.nonzero(as_tuple=False).flatten()
            action = valid_actions[torch.randint(len(valid_actions), (1,))].view(1, 1)

        return action

    def plot_durations(self, averaging_window=50, title="", path=None):
        averages = []
        for i in range(1, len(self.episode_durations) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_durations[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with " + str(averaging_window) + "-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(self.episode_durations)), self.episode_durations, s=2)
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    import matplotlib.pyplot as plt

    def plot_rewards(self, averaging_window=50, title="Rewards over Time", path=None):
        averages = []
        for i in range(1, len(self.episode_rewards) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_rewards[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(title)
        plt.plot(averages, color="blue")
        plt.scatter(range(len(self.episode_rewards)), self.episode_rewards, s=2)
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
        
    def save_training_log(self, prefix):
        df = pd.DataFrame({"episode_rewards": self.episode_rewards, "episode_durations": self.episode_durations})
        df.to_csv(f"{prefix}_training_log.csv")
        self.plot_durations(title="Deep Q Learning", path=f"{prefix}_durations.png")
        self.plot_rewards(title="Deep Q Learning", path=f"{prefix}_rewards.png")

    def learn(
        self,
        num_episodes=500,
        batch_size=64,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        target_net_update_rate=0.005,
        learning_rate=1e-4,
        opponents=None,
        visualize=False,
        early_stopping_condition=None,
    ):
        optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
        expo_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup_scheduler, expo_scheduler])
        steps = 0
        self.episode_rewards = []  # List to store rewards for each episode
        self.episode_durations = []  # List to store durations for each episode

        for i_episode in range(num_episodes):
            self.env.reset()

            # 50% probability to place a -1 on the field
            if random.random() < 0.5:
                self.env.player = -1
                if opponents is None:
                    opponent = None
                else:
                    selection_pool = opponents + [None]
                    opponent = random.choice(selection_pool)
                if opponent is None:
                    self.env._random_moove()
                else:
                    opponent.play(self.env)

            state = self.get_state()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            episode_reward = 0

            for t in count():
                eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
                    -1.0 * steps / eps_decay
                )
                steps += 1

                action = self._eps_greedy_action(state, eps_threshold)
                
                # Check the shape of action
                # print(f"Action shape: {action.shape}")

                try:
                    self.env.moove(
                        (action.item() // self.size, action.item() % self.size)
                    )
                except:
                    print(
                        f"Action_values: {self.policy_net(state).detach().cpu().numpy()}"
                    )
                    print(f"Action: {action}")
                    print(f"Board: {self.env.board}")

                if self.env.winner == 0:
                    if opponents is None:
                        opponent = None
                    else:
                        selection_pool = opponents + [None]
                        opponent = random.choice(selection_pool)
                    if opponent is None:
                        self.env._random_moove()
                    else:
                        opponent.play(self.env)

                done = self.env.winner != 0
                reward = torch.tensor(
                    [1 if self.env.winner == 1 else 0],
                    device=self.device,
                )
                episode_reward += reward.item()  # Accumulate reward

                next_state = (
                    None
                    if done
                    else torch.tensor(
                        self.get_state(), dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )

                self.memory.save(state, action, next_state, reward)
                state = next_state

                if len(self.memory) >= batch_size:
                    transitions = self.memory.sample(batch_size)
                    batch = transitionData(*zip(*transitions))

                    non_final_mask = torch.tensor(
                        tuple(map(lambda s: s is not None, batch.next_state)),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    non_final_next_states = torch.cat(
                        [s for s in batch.next_state if s is not None]
                    )

                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    
                    state_action_values = self.policy_net(state_batch).gather(
                        1, action_batch
                    )
                    
                    next_state_values = torch.zeros(batch_size, device=self.device)
                    with torch.no_grad():
                        next_state_values[non_final_mask] = self.target_net(
                            non_final_next_states
                        ).max(1)[0]

                    expected_state_action_values = (
                        next_state_values * gamma
                    ) + reward_batch
                    
                    valid_mask = (state_batch.round().int() == 0).gather(
                        1, action_batch
                    ).squeeze(1)
                    expected_state_action_values = expected_state_action_values * valid_mask.float()
                    criterion = nn.BCELoss(weight=valid_mask.float().unsqueeze(1))
                    loss = criterion(
                        state_action_values, expected_state_action_values.unsqueeze(1)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                    optimizer.step()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * target_net_update_rate + target_net_state_dict[key] * (
                        1 - target_net_update_rate
                    )
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(
                        episode_reward
                    )  # Store the total reward for this episode
                    break

            print(
                f"\rEpisode {i_episode} of {num_episodes}: Avg. Reward: {np.mean(self.episode_rewards[-50:]):.3f}, Avg. Duration: {np.mean(self.episode_durations[-50:]):.3f}",
                end="",
            )
            if early_stopping_condition and np.mean(self.episode_rewards[-50:]) >= early_stopping_condition and i_episode > 50:
                print()
                print("Solved in", i_episode, "episodes!")
                break
            scheduler.step()
            # if i_episode % 200 == 0:
            #     print(f"Episode {i_episode} of {num_episodes}")
            #     #self.plot_rewards()
        print()
        print("Complete")

        if visualize:
            self.plot_durations(title="Deep Q Learning")
            self.plot_rewards()


if __name__ == "__main__":
    folder = "deepq_models/deepq_transformers_7x7_5"
    #init_model = f"{folder}/deepq_hex_19.pth"
    init_model = None
    size = 7
    learning_rate = 1e-4
    gamma = 0.9
    base_iterations = 2000
    iterations_increment = 200
    visualise = False
    early_stopping_condition = 1.0

    if "deepq_hex_base.pth" not in os.listdir(folder):
        agent = deepQ(size=size, memory_length=10_000)
        agent.initialize_networks()

        print("Training the base model against random opponents...")
        agent.learn(
            num_episodes=base_iterations if torch.cuda.is_available() else 200,
            visualize=visualise,
            learning_rate=learning_rate,
            gamma=gamma,
            early_stopping_condition=early_stopping_condition,
        )

        agent.save(f"{folder}/deepq_hex_base.pth")
        agent.save_training_log(f"{folder}/deepq_hex_base")
    
    agent = deepQ(size=size, memory_length=10_000)
    agent.initialize_networks()
        
    if init_model:
        agent.policy_net.load_state_dict(torch.load(init_model))
        agent.target_net.load_state_dict(torch.load(init_model))

    for i in range(0, 50):
        trained_agents = []

        pth_files = list(Path(folder).glob("*.pth"))
        for file in pth_files:
            trained_agent = HexAgent(model_path=str(file), size=size)
            trained_agents.append(trained_agent)
        
        #agent = deepQ(size=size, memory_length=10_000)
        #agent.initialize_networks()
        episodes = base_iterations + i * iterations_increment
        agent.learn(
            num_episodes=episodes if torch.cuda.is_available() else 200,
            opponents=trained_agents,
            visualize=visualise,
            learning_rate=learning_rate,
            gamma=gamma,
            early_stopping_condition=early_stopping_condition,
        )

        if np.mean(agent.episode_rewards[-50:]) > 0.5:
            agent.save(f"{folder}/deepq_hex_{i}.pth")
            agent.save_training_log(f"{folder}/deepq_hex_{i}")
        else:
            i -= 1
