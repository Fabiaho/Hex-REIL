from __future__ import annotations
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from conv_agent import BaseAgent, ConvAgent, RandomAgent
from hex_engine import hexPosition

folder_5x5 = Path("deepq_models/deepq_convolutional_5x5")
folder_7x7 = Path("deepq_models/deepq_convolutional_7x7_3")


def match(engine: hexPosition, agent1: BaseAgent, agent2: BaseAgent) -> tuple[int, int]:
    score1 = score2 = 0
    engine.reset()
    agent1.player = 1
    agent2.player = -1
    engine.machine_vs_machine(agent1, agent2, interactive=False)
    score1 += engine.winner == 1
    score2 += engine.winner == -1
    engine.reset()
    agent1.player = -1
    agent2.player = 1
    engine.machine_vs_machine(agent2, agent1, interactive=False)
    score1 += engine.winner == -1
    score2 += engine.winner == 1
    return score1, score2	

best_models = {}

model_names = [f"deepq_hex_{i}.pth" for i in ["base"] + list(range(0, 49))]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
for folder, size, ax in zip([folder_5x5, folder_7x7], [5, 7], axs):
    agents = [ConvAgent(folder / model_name, size=size) for model_name in model_names]
    scores = np.zeros(len(agents), dtype=int)
    engine = hexPosition(size)
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1 = agents[i]
            agent2 = agents[j]
            score1, score2 = match(engine, agent1, agent2)
            scores[i] += score1
            scores[j] += score2
    print(f"Scores for {folder}: {scores}")
    ax.bar(range(len(agents)), scores)
    ax.set_xlabel("Model Generation")
    ax.set_title(f"Board size {size}$\\times${size}")
    best_models[size] = [agents[k] for k in np.argsort(scores)[::-1][:10]]
    print(f"Best models for {folder}: {best_models}")

axs[0].set_ylabel("Wins")
axs[1].set_yticks([])
fig.suptitle("Within-Group scores for different models")
plt.savefig("term_paper/board_size_generalization.pdf")



num_matches_against_random = 20

for board_size in [5, 7]:
    engine = hexPosition(board_size)
    
    agents_5x5 = best_models[5]
    agents_7x7 = best_models[7]
    for agent in agents_5x5 + agents_7x7:
        agent.size = board_size
    random_agent = RandomAgent(size=board_size)
    
    # Play against random agent
    total_score5 = total_scoreR = 0
    for agent_5x5 in agents_5x5:
        print(f"Comparing {agent_5x5.path} with random agent on board size {board_size}x{board_size} ... ", end="")
        score5 = scoreR = 0
        for _ in range(num_matches_against_random):
            s5, sR = match(engine, agent_5x5, random_agent)
            score5 += s5
            scoreR += sR
        total_score5 += score5
        total_scoreR += scoreR
        print(f"Score: {score5} - {scoreR}")
    print(f"Total score: {total_score5} - {total_scoreR}")
        
    # Play against random agent
    total_score5 = total_scoreR = 0
    for agent_7x7 in agents_7x7:
        print(f"Comparing {agent_7x7.path} with random agent on board size {board_size}x{board_size} ... ", end="")
        score5 = scoreR = 0
        for _ in range(num_matches_against_random):
            s5, sR = match(engine, agent_5x5, random_agent)
            score5 += s5
            scoreR += sR
        total_score5 += score5
        total_scoreR += scoreR
        print(f"Score: {score5} - {scoreR}")
    print(f"Total score: {total_score5} - {total_scoreR}")
    
    # Play against other trained agents
    total_score5 = total_score7 = 0
    for agent_5x5 in agents_5x5:
        for agent_7x7 in agents_7x7:
            print(f"Comparing {agent_5x5.path} with {agent_7x7.path} on board size {board_size}x{board_size} ... ", end="")
            score5, score7 = match(engine, agent_5x5, agent_7x7)
            print(f"Score: {score5} - {score7}")
            total_score5 += score5
            total_score7 += score7
    print(f"Total score: {total_score5} - {total_score7}")
