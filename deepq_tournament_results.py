import pandas as pd
import re
import matplotlib.pyplot as plt

# Load the results
results = pd.read_csv('deepq_tournament_results_4.csv')



fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6, 6), sharex=True)
markers = ["*", "x", "+", ".", "_"]

for i, architecture in enumerate(["1-feedforward", "4-convpos", "2-convolutional", "3-transformer", "5-transformer2"]):
    data = results[results["architecture"] == architecture]
    ax[0].scatter(
        x=data["training_rank"], 
        y=data["group_score"],
        marker=markers[i],
    )
    ax[1].scatter(
        x=data["training_rank"],
        y=data["total_score"],
        marker=markers[i],
    )
    
ax[0].set_ylabel("Within-group score")
ax[0].set_xticks([])

ax[1].set_ylabel("Overall score")
ax[1].set_xlabel("Model Generation")
ax[1].set_xticks(range(0, 51, 10))

average_score = results["total_score"].mean()
average_games_played = average_score * 2
print(f"Maximum score: {results['total_score'].max()}")
print(f"Average score: {average_score}")

fig.suptitle("Tournament results")
fig.legend(["Feedforward", "Convolutional+", "Convolutional", "Transformer", "Transformer2"], loc='lower right')
plt.tight_layout()
plt.savefig("term_paper/tournament_results.pdf")
plt.show()