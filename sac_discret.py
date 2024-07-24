from collections import deque
from itertools import count
from typing import Deque, NamedTuple
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

from hex_engine import hexPosition


class SAC_Agent:
    def __init__(self, field_size: int = 7, discount_rate: float = 0.9, training_episodes_per_eval_episode: int = 10, device: str = "cuda"):
        self.field_size = field_size
        self.discount_rate = discount_rate
        self.training_episodes_per_eval_episode = training_episodes_per_eval_episode
        self.device = device
        self.eps = torch.tensor([1e-8], device=self.device)
        self.learning_rate = 1e-3
        self.gradient_clipping_norm = None
        # Init all the estimators
        self.critic_1 = Critic(field_size).to(self.device)
        self.critic_2 = Critic(field_size).to(self.device)
        self.critic_target_1 = Critic(field_size).to(self.device)
        self.critic_target_2 = Critic(field_size).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.actor = Actor(field_size).to(self.device)
        # Init the optimizers
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=self.learning_rate)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=self.learning_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        # Automatic entropy tuning
        self.target_entropy = -np.log((1.0 / self.field_size**2)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate, eps=1e-4)
        
    def produce_action_and_action_info(self, state: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        # Get the action from the actor
        valid_actions = state.round().int() == 0
        action_probabilities = self.actor(state)
        # Mask out the invalid actions and renormalize
        action_probabilities = action_probabilities * valid_actions
        action_probabilities = action_probabilities / torch.max(action_probabilities.sum(), self.eps)
        
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        log_action_probabilities = torch.log(torch.max(action_probabilities, self.eps))
        return action, (action_probabilities, log_action_probabilities), max_probability_action
        
    def calculate_critic_losses(self, state_batch: Tensor, action_batch: Tensor, reward_batch: Tensor, next_state_batch: Tensor, mask_batch: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target_1(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + mask_batch.float() * self.discount_rate * (min_qf_next_target)

        qf1 = self.critic_1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss
    
    def take_optimisation_step(self, optimizer: torch.optim.Optimizer, network: nn.Module, loss: Tensor, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients
        
    def soft_update_of_target_network(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def update_critic_parameters(self, critic_loss_1: Tensor, critic_loss_2: Tensor, target_update_rate: float):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_1_optimizer, self.critic_1, critic_loss_1, self.gradient_clipping_norm)
        self.take_optimisation_step(self.critic_2_optimizer, self.critic_2, critic_loss_2, self.gradient_clipping_norm)
        self.soft_update_of_target_network(self.critic_1, self.critic_target_1, target_update_rate)
        self.soft_update_of_target_network(self.critic_2, self.critic_target_2, target_update_rate)
    
    def calculate_actor_loss(self, state_batch: Tensor) -> tuple[Tensor, Tensor]:
        _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_1(state_batch)
        qf2_pi = self.critic_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
    
    def calculate_entropy_tuning_loss(self, log_pi: Tensor) -> Tensor:
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss
    
    def update_actor_parameters(self, actor_loss: Tensor, alpha_loss: Tensor):
        self.take_optimisation_step(self.actor_optimizer, self.actor, actor_loss, self.gradient_clipping_norm)
        self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
        self.alpha = self.log_alpha.exp()

    
    def get_state(self, env: hexPosition) -> Tensor:
        return torch.tensor(
            np.asarray(env.board).flatten(), 
            device=self.device
        ).unsqueeze(0).float()
    
    def learn(self, max_episodes: int, env: hexPosition, batch_size: int = 32, target_net_update_rate=0.005) -> tuple[list[float], list[int]]:
        replay_memory = ReplayMemory(10000)
        episode_rewards = []
        episode_durations = []
        for episode in range(max_episodes):
            env.reset()
            episode_reward = 0.0
            state = self.get_state(env)
            # Normally one would use a fixed number of steps, but in our case it is simpler to just
            # stop when the game is over
            for step in count():
                # Sample an action
                action, _, _ = self.produce_action_and_action_info(state)
                try:
                    env.moove(
                        (action.item() // self.field_size, action.item() % self.field_size)
                    )
                except:
                    print(f"Action: {action}")
                    print(f"Board: {env.board}")
                if env.winner == 0:
                    env._random_moove()
                done = env.winner != 0
                reward = torch.tensor(
                    [1 if env.winner == 1 else 0],
                    device=self.device,
                )
                episode_reward += reward.item()  # Accumulate reward
                next_state = None if done else self.get_state(env)
                replay_memory.save(state, action, next_state, reward)
                state = next_state
                
                if done:
                    episode_durations.append(step + 1)
                    episode_rewards.append(episode_reward)
                    break
            print(
                f"\rEpisode {episode} of {max_episodes}: Avg. Reward: {np.mean(episode_rewards[-50:]):.3f}, Avg. Duration: {np.mean(episode_durations[-50:]):.3f}",
                end="",
            )
            for training_step in range(self.training_episodes_per_eval_episode):
                batch_transitions = replay_memory.sample(batch_size)
                batch = TransitionData(*zip(*batch_transitions))
                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=self.device,
                    dtype=torch.bool,
                ).unsqueeze(1)
                default_state = torch.zeros(1, self.field_size**2, device=self.device)
                next_states_batch = torch.cat(
                    [s if s is not None else default_state for s in batch.next_state]
                )
                state_batch = torch.cat(batch.state)
                action_batch = torch.stack(batch.action)
                reward_batch = torch.stack(batch.reward)
                
                critic_1_loss, critic_2_loss = self.calculate_critic_losses(
                    state_batch, action_batch, reward_batch, next_states_batch, non_final_mask
                )
                self.update_critic_parameters(critic_1_loss, critic_2_loss, target_net_update_rate)
                
                actor_loss, log_action_probabilities = self.calculate_actor_loss(state_batch)
                alpha_loss = self.calculate_entropy_tuning_loss(log_action_probabilities)
                self.update_actor_parameters(actor_loss, alpha_loss)
            
        print()
        print("Complete")
        return episode_rewards, episode_durations


class BaseFeedForwardNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int):
        super(BaseFeedForwardNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)

class Critic(BaseFeedForwardNN):
    def __init__(self, field_size, hidden_size:int = 256, hidden_layers:int = 2):
        super(Critic, self).__init__(field_size ** 2, hidden_size, field_size ** 2, hidden_layers)
    def forward(self, x):
        x = super().forward(x)
        return nn.Sigmoid()(x)
        
        
class Actor(BaseFeedForwardNN):
    def __init__(self, field_size, hidden_size:int = 256, hidden_layers:int = 2):
        super(Actor, self).__init__(field_size ** 2, hidden_size, field_size ** 2, hidden_layers)
    def forward(self, x):
        x = super().forward(x)
        return nn.Softmax(dim=1)(x)


class TransitionData(NamedTuple):
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor


class ReplayMemory(object):
    """
    Store transitions consisting of 'state', 'action', 'next_state', 'reward'.
    """

    def __init__(self, length: int):
        self.memory: Deque[TransitionData] = Deque(maxlen=length)

    def save(self, state, action, next_state, reward):
        self.memory.append(TransitionData(state, action, next_state, reward))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.memory), batch_size, replace=True)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)
    
    
def plot_results(episode_rewards, episode_durations):
    import matplotlib.pyplot as plt
    t = np.arange(len(episode_rewards))
    avg_rewards = [np.mean(episode_rewards[max(0, i-50):i+1]) for i in range(len(episode_rewards))]
    avg_durations = [np.mean(episode_durations[max(0, i-50):i+1]) for i in range(len(episode_durations))]
    fig, ax1 = plt.subplots()
    ax2 = plt.gca().twinx()
    ax2.plot(t, avg_durations, 'r--', linewidth=0.5)
    ax1.plot(t, avg_rewards, 'k', linewidth=2)
    ax1.set_ylim([0, 1])
    ax1.set_title("SAC Discrete Hex")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax2.set_ylabel('Duration', color='r')
    plt.show()
 
    
    
if __name__ == "__main__":
    board_size = 4
    env = hexPosition(board_size)
    agent = SAC_Agent(board_size)
    eposide_rewards, episode_durations = agent.learn(2000, env)
    plot_results(eposide_rewards, episode_durations)