import matplotlib.pyplot as plt
from typing import List

from src.nn.memory_data import MemoryData

def plot_batch_win_percentage(memory: MemoryData):
    """
    Plot first player win percentage per batch and overlay epsilon progression per batch.
    """
    import matplotlib.pyplot as plt

    batch_winners = memory.batch_winners
    batch_size = memory.batch_size
    epsilon_prog = memory.epsilon_progression

    batches = list(range(1, len(batch_winners) + 1))
    first_player_win_percentages = []
    for winner_dict in batch_winners:
        first_id = list(winner_dict.keys())[0]
        total = sum(winner_dict.values())
        wins = winner_dict.get(first_id, 0)
        pct = (wins / total) * 100 if total > 0 else 0
        first_player_win_percentages.append(pct)

    # Epsilon updated once per batch; use stored progression entries per batch
    batch_epsilon = epsilon_prog[:len(batches)]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:blue'
    ax1.plot(batches, first_player_win_percentages, marker='o', color=color1, label='First Player Win %')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Win Percentage (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(batches, batch_epsilon, marker='x', linestyle='--', color=color2, label='Epsilon')
    ax2.set_ylabel('Epsilon', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('First Player Win Percentage and Epsilon per Batch')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
