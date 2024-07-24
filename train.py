import gymnasium as gym
from custom_hex_env import CustomHexEnv  # Ensure this registers the custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np


# Define a function to create the environment
def make_env(opponent_type="self", opponent_model=None):
    return CustomHexEnv(render_mode="rgb_array", size=5, opponent_type=opponent_type, opponent_model=opponent_model)

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)


##############################
# TRAIN MODEL 1 SELF TRAINING
##############################
# Create the environment for self-play
env = Monitor(make_env(opponent_type="self"), log_dir)
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

check_env(env, warn=True)
obs, _ = env.reset()


# Create and train the PPO model
model_1 = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model_1.learn(total_timesteps=100_000)
model_1.save("model_1")

# Plotting training progress
log_data = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
log_data["Episode"] = log_data["l"].cumsum()
episode_rewards = log_data.groupby("Episode")["r"].sum().reset_index()
episode_rewards["mean_reward"] = episode_rewards["r"].rolling(window=1000).mean()
plt.plot(episode_rewards["Episode"], episode_rewards["mean_reward"])
plt.xlabel("Episodes")
plt.ylabel("Mean Reward (last 1000 episodes)")
plt.title("Mean Reward per Episode")
plt.show()


##############################
# TRAIN MODEL 2 COMPETE AGAINST MODEL 1
##############################
# Create the environment where model_1 is the opponent

env = Monitor(make_env(opponent_type="trained", opponent_model=model_1), log_dir)
check_env(env, warn=True)
obs, _ = env.reset()

# Create and train the second PPO model
model_2 = PPO("MlpPolicy", env, verbose=1)
model_2.learn(total_timesteps=100_000)
model_2.save("model_2")

##############################



##############################
#LOAD MODELS AND PLAY AGAINST EACH OTHER
##############################


# Load the trained models
model_1 = MaskablePPO.load("model_1")
model_2 = PPO.load("model_2")

# Custom environment for comparison
class CompEnv(CustomHexEnv):
    def __init__(self, model_1, model_2):
        super().__init__(render_mode="human", size=5, opponent_type="trained", opponent_model=model_2)
        self.model_1 = model_1
        self.model_2 = model_2
        self.current_model = self.model_1  # Model 1 starts

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        if not done:
            self.current_model = self.model_2 if self.current_model == self.model_1 else self.model_1
            opponent_action, _ = self.current_model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = super().step(opponent_action)
            done = terminated or truncated
        return obs, reward, terminated, truncated, info

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

comp_env = Monitor(CompEnv(model_1, model_2), log_dir)

# Evaluate the models by playing a series of games
n_games = 10  # Reduce the number of games for debugging
results = {'model_1_wins': 0, 'model_2_wins': 0}

for i in range(n_games):
    obs, _ = comp_env.reset()
    done = False
    current_model = model_1  # Model 1 starts each game
    print(f"Game {i + 1}")
    step_count = 0  # Add step counter for debugging
    while not done:
        step_count += 1
        action, _ = current_model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = comp_env.step(action)
        comp_env.render()
        done = terminated or truncated
        if done:
            if reward > 0:
                results['model_1_wins'] += 1
            elif reward < 0:
                results['model_2_wins'] += 1
            print(f"Game {i + 1} ended in {step_count} steps.")
            break
        current_model = model_2 if current_model == model_1 else model_1

print(f"Results after {n_games} games:")
print(f"Model 1 wins: {results['model_1_wins']}")
print(f"Model 2 wins: {results['model_2_wins']}")