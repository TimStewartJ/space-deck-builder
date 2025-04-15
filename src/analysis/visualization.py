import matplotlib.pyplot as plt
from typing import List

def calculate_epsilon_values(num_episodes, initial_epsilon=0.995, decay_interval=5):
    """Calculate epsilon values for each episode based on the decay schedule."""
    epsilon_values = []
    current_epsilon = initial_epsilon

    for episode in range(num_episodes):
        epsilon_values.append(current_epsilon)
        if (episode + 1) % decay_interval == 0:
            current_epsilon *= initial_epsilon

    return epsilon_values

def plot_win_rate_and_epsilon(episode_outcomes: List[int], chunk_size=100, initial_epsilon=0.995, decay_interval=5):
    """Plot win rate over time and epsilon decay rate."""
    chunks = []
    rates = []

    for i in range(0, len(episode_outcomes), chunk_size):
        chunk = episode_outcomes[i:i+chunk_size]
        if chunk:
            win_rate = sum(chunk) / len(chunk)
            chunk_num = i // chunk_size
            chunks.append(chunk_num)
            rates.append(win_rate)

    epsilon_values = calculate_epsilon_values(len(episode_outcomes), initial_epsilon, decay_interval)
    chunk_epsilon_values = [epsilon_values[min(i*chunk_size, len(epsilon_values)-1)] for i in chunks]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode Chunks (each represents {} episodes)'.format(chunk_size))
    ax1.set_ylabel('Win Rate', color=color)
    ax1.plot(chunks, rates, marker='o', linestyle='-', color=color, label='Win Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim((0.0, 1.05))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Epsilon Value', color=color)
    ax2.plot(chunks, chunk_epsilon_values, marker='x', linestyle='--', color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0.0, 1.05))

    plt.title('Win Rate and Exploration Epsilon Over Time')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.show()
