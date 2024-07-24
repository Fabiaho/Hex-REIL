from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

architectures = ["Feedforward", "Convolutional+", "Convolutional", "Transformer", "Transformer2"]

folders = [
    Path("deepq_models/deepq_feedforward_7x7_2"),
    Path("deepq_models/deepq_convpos_7x7"),
    Path("deepq_models/deepq_convolutional_7x7_3"),
    Path("deepq_models/deepq_transformers_7x7_4"),
    Path("deepq_models/deepq_transformers_7x7_5"),
]

protocols = [
    folder / "deepq_hex_base_training_log.csv" for folder in folders
]

fig = plt.figure(figsize=(8, 4))
for protocol, architecture in zip(protocols, architectures):
    df = pd.read_csv(protocol, index_col=0)
    rewards = df["episode_rewards"]
    mean_rewards = rewards.rolling(window=50).mean()
    line = plt.plot(mean_rewards, label=f"{architecture} ({len(mean_rewards)} episodes)")
    color = line[0].get_color()
    plt.scatter(len(mean_rewards) - 1, mean_rewards.iloc[-1], color=color)

plt.ylim(0.5, 1.05)
plt.grid()
plt.legend()
plt.ylabel("Mean reward (50 episodes)")
plt.xlabel("Episode")
plt.title("Training progress for different architectures against random agent")
plt.savefig("term_paper/architecture_training_progress.pdf")
plt.show()
