# deepq_hex.py

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
            hidden=1,
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
        recoded_action = self.recode_action(action)
        env.moove((recoded_action // self.size, recoded_action % self.size))

    def recode_action(self, action):
        row = action // self.size
        col = action % self.size
        recoded_row = self.size - 1 - col
        recoded_col = self.size - 1 - row
        recoded_action = recoded_row * self.size + recoded_col
        return recoded_action


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


class FeedForwardPolicy(nn.Module):
    def __init__(self, input_length: int, width: int, output_length: int, hidden=1):
        super(FeedForwardPolicy, self).__init__()
        self.linear1 = nn.Linear(input_length, width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(hidden)]
        )
        self.output_layer = nn.Linear(width, output_length)
        self.eps = torch.tensor(1e-6, dtype=torch.float32)

    def forward(self, x):
        valid_actions = (x.round().int() == 0).float()
        x = self.linear1(x)
        x = nn.ReLU()(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.output_layer(x)
        x = nn.Softmax(dim=1)(x) * valid_actions
        x = x / torch.maximum(x.sum(1, keepdim=True), self.eps)
        return x


def make_torch_net(input_length: int, width: int, output_length: int, hidden=1):
    net = FeedForwardPolicy(input_length, width, output_length, hidden)
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

    def get_state(self):
        return np.array(self.env.board).flatten()

    def initialize_networks(self, width=128, hidden=1):
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

    def plot_durations(self, averaging_window=50, title=""):
        averages = []
        for i in range(1, len(self.episode_durations) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_durations[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with " + str(averaging_window) + "-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(self.episode_durations)), self.episode_durations, s=2)
        plt.show()

    import matplotlib.pyplot as plt

    def plot_rewards(self, averaging_window=50, title="Rewards over Time"):
        averages = []
        for i in range(1, len(self.episode_rewards) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_rewards[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(title)
        plt.plot(averages, color="blue")
        plt.scatter(range(len(self.episode_rewards)), self.episode_rewards, s=2)
        plt.show()

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

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
    ):
        optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        steps = 0
        self.episode_rewards = []  # List to store rewards for each episode

        for i_episode in range(num_episodes):
            self.env.reset()

            # 50% probability to place a -1 on the field
            if random.random() < 0.5:
                self.env.player = -1
                self.env._random_moove()

            state = self.get_state()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            episode_reward = 0

            if opponents is None:
                opponent = None
            else:
                selection_pool = opponents + [None]
                opponent = random.choice(selection_pool)

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
                    if opponent is None:
                        self.env._random_moove()
                    else:
                        opponent.play(self.env)

                done = self.env.winner != 0
                reward = torch.tensor(
                    [1 if self.env.winner == 1 else -1 if self.env.winner == -1 else 0],
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

                    criterion = nn.SmoothL1Loss()
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
            if np.mean(self.episode_rewards[-50:]) == 1 and i_episode > 50:
                print()
                print("Solved in", i_episode, "episodes!")
                break
            # if i_episode % 200 == 0:
            #     print(f"Episode {i_episode} of {num_episodes}")
            #     #self.plot_rewards()
        print()
        print("Complete")

        if visualize:
            self.plot_durations(title="Deep Q Learning")
            self.plot_rewards()


if __name__ == "__main__":
    folder = "deepq_models"
    size = 5

    if "deepq_hex_base.pth" not in os.listdir(folder):
        agent = deepQ(size=size, memory_length=10_000)
        agent.initialize_networks()

        print("Training the base model against random opponents...")
        agent.learn(
            num_episodes=3_000 if torch.cuda.is_available() else 200,
            visualize=True,
        )

        agent.save(f"{folder}/deepq_hex_base.pth")

    for i in range(10):
        trained_agents = []

        all_files = os.listdir(folder)
        # Filter the list to include only .pth files
        pth_files = [file for file in all_files if file.endswith(".pth")]

        for file in pth_files:
            trained_agent = HexAgent(model_path=f"{folder}/{file}", size=size)
            trained_agents.append(trained_agent)

        agent = deepQ(size=size, memory_length=10_000)
        agent.initialize_networks()
        episodes = 3_000 + i * 1_000
        agent.learn(
            num_episodes=episodes if torch.cuda.is_available() else 200,
            opponents=trained_agents,
            visualize=True,
        )

        if np.mean(agent.episode_rewards[-50:]) > 0.5:
            agent.save(f"deepq_models/deepq_hex_{i}.pth")
        else:
            i -= 1
