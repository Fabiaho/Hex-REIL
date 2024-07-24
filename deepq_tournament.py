from pathlib import Path
from typing import Callable, NamedTuple, Type
from conv_agent import BaseAgent, ConvPosAgent, FeedForwardAgent, ConvAgent, TransformerAgent, TransformerAgent2
from deepq_hex_feedforward import FeedForwardPolicy
from deepq_hex_convolutional import ConvolutionalPolicy
from deepq_hex_transformer import TransformerPolicy
from hex_engine import hexPosition
import torch
from torch import nn
from numpy.typing import NDArray
import numpy as np
from csv import DictWriter
import re
import pandas as pd

model_dirs = {
    "feedforward": Path("deepq_feedforward_7x7_3"),
    "convolutional": Path("deepq_convolutional_7x7_3"),
    "transformer": Path("deepq_transformers_7x7_4"),
    "convpos": Path("deepq_convpos_7x7"),
    "transformer2": Path("deepq_transformers_7x7_5"),
}
model_classes = {
    "feedforward": FeedForwardAgent,
    "convolutional": ConvAgent,
    "transformer": TransformerAgent,
    "convpos": ConvPosAgent,
    "transformer2": TransformerAgent2,

}
tournament_log = Path("deepq_tournament_results_4.csv")


def get_type(params: str):
    if "feedforward" in params:
        return "1-feedforward"
    elif "convolutional" in params:
        return "2-convolutional"
    elif "transformers_7x7_5" in params:
        return "5-transformer2"
    elif "transformer" in params:
        return "3-transformer"
    elif "convpos" in params:
        return "4-convpos"
    raise ValueError(f"Unknown type for params: {params}")

def get_training_rank(params: str):
    if "deepq_hex_base" in params:
        return 0
    match = re.match(r".+deepq_hex_(\d+).pth", params)
    if not match:
        raise ValueError(f"Unknown params: {params}")
    return int(match.group(1)) + 1


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


class ModelInfo():
    
    def __init__(self, clazz: Type[BaseAgent], model_path: Path):
        self.model = clazz(model_path, size=7)
        self.model_path = model_path
        self.architecture = get_type(str(model_path))
        self.training_rank = get_training_rank(str(model_path))
        
    def __repr__(self):
        return f"{self.architecture} {self.training_rank}"

model_files = {
    architecture: list(model_dir.glob("*.pth")) for architecture, model_dir in model_dirs.items()
}
model_infos = {
    architecture: [ModelInfo(model_classes[architecture], model_file) for model_file in model_files[architecture]] for architecture in model_files
}
for architecture in model_infos:
    model_infos[architecture].sort(key=lambda model_info: model_info.training_rank)

all_model_infos = [model_info for model_infos in model_infos.values() for model_info in model_infos]
group_scores = np.zeros(len(all_model_infos), dtype=int)
total_scores = np.zeros(len(all_model_infos), dtype=int)

for i, model_info1 in enumerate(all_model_infos):
    for j, model_info2 in enumerate(all_model_infos[i + 1:]):
        j += i + 1
        print(f"Match {model_info1} vs {model_info2}: ", end="")
        score1, score2 = match(hexPosition(7), model_info1.model, model_info2.model)
        total_scores[i] += score1
        total_scores[j] += score2
        if model_info1.architecture == model_info2.architecture:
            group_scores[i] += score1
            group_scores[j] += score2
        print(f"{score1}-{score2}")
        
final_results = pd.DataFrame({
    "model_params": [str(model_info.model_path) for model_info in all_model_infos],
    "architecture": [model_info.architecture for model_info in all_model_infos],
    "training_rank": [model_info.training_rank for model_info in all_model_infos],
    "total_score": total_scores,
    "group_score": group_scores,
})
final_results.sort_values(by=["architecture", "training_rank"], inplace=True)
final_results.to_csv(tournament_log, index=False)
