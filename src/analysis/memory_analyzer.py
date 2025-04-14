from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.utils.logger import log
from collections import Counter
from src.engine.actions import Action, ActionType

def load_memory(memory_file_path: Path = Path("memory.pkl")) -> List[List[Tuple]]:
    """Loads the agent's memory from a pickle file."""
    try:
        with open(memory_file_path, 'rb') as f:
            memory = pickle.load(f)
        log(f"Successfully loaded {len(memory)} episodes from {memory_file_path}.")
        return memory
    except FileNotFoundError:
        log(f"Error: Memory file not found at {memory_file_path}")
        return []
    except Exception as e:
        log(f"Error loading memory from {memory_file_path}: {e}")
        return []

def analyze_win_loss_and_steps(memory: List[List[Tuple]]) -> Tuple[List[int], List[int], Counter]:
    """
    Analyzes the memory to determine win/loss outcomes, steps per episode,
    and the frequency of different action types.
    Assumes the NeuralAgent is Player 1 and a positive reward at the end signifies a win.

    Args:
        memory: A list of episodes, where each episode is a list of transitions.

    Returns:
        A tuple containing:
        - A list of integers representing outcomes (1 for win, 0 for loss).
        - A list of integers representing the number of steps in each valid episode.
        - A Counter object mapping ActionType to its frequency.
    """
    outcomes = []
    steps_per_episode = []
    action_counts = Counter()

    for i, episode in enumerate(memory):
        if not episode:
            log(f"Warning: Episode {i} is empty.")
            continue

        num_steps = 0
        for transition in episode:
            state, action, reward, next_state, done = transition
            if isinstance(action, Action): # Ensure action is an Action object
                action_counts[action.type] += 1
            else:
                # Handle cases where action might not be an Action object (e.g., if stored differently)
                log(f"Warning: Encountered unexpected action format in episode {i}, step {num_steps}: {action}")
            num_steps += 1
            
        # Assuming the last reward determines win/loss for the episode
        last_reward = episode[-1][2] if episode[-1][2] is not None else 0
        if last_reward > 0:
            outcomes.append(1)  # Win
        else:
            outcomes.append(0)  # Loss or Draw
        steps_per_episode.append(num_steps)

    log(f"Analyzed {len(outcomes)} completed episodes for outcomes and steps.")
    log(f"Total actions analyzed: {sum(action_counts.values())}")
    return outcomes, steps_per_episode, action_counts

def calculate_win_rate_over_time(outcomes: List[int], chunk_size: int = 100) -> Tuple[List[int], List[float]]:
    """
    Calculates the win rate over chunks of episodes.

    Args:
        outcomes: List of win/loss outcomes (1 for win, 0 for loss).
        chunk_size: The number of episodes per chunk for calculating win rate.

    Returns:
        A tuple containing:
        - List of episode counts (end of each chunk).
        - List of win rates corresponding to each chunk.
    """
    if not outcomes:
        return [], []

    episode_chunks = []
    win_rates = []
    total_episodes = len(outcomes)

    for i in range(chunk_size, total_episodes + chunk_size, chunk_size):
        chunk_end = min(i, total_episodes)
        current_chunk_outcomes = outcomes[:chunk_end]
        if not current_chunk_outcomes:
            continue
            
        win_rate = np.mean(current_chunk_outcomes) * 100  # Calculate win rate as percentage
        episode_chunks.append(chunk_end)
        win_rates.append(win_rate)
        log(f"Episodes {1}-{chunk_end}: Win Rate = {win_rate:.2f}%")

    return episode_chunks, win_rates

def plot_win_rate(episode_chunks: List[int], win_rates: List[float]):
    """Plots the win rate over time."""
    if not episode_chunks or not win_rates:
        log("No data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(episode_chunks, win_rates, marker='o')
    plt.title('Neural Agent Win Rate Over Training Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.ylim(0, 100) # Win rate is a percentage
    # Ensure x-axis starts near 0 or the first chunk size
    plt.xlim(left=0) 
    
    # Add text for final win rate
    if win_rates:
        final_win_rate = win_rates[-1]
        final_episodes = episode_chunks[-1]
        plt.text(final_episodes, final_win_rate, f'{final_win_rate:.1f}%', 
                 ha='right', va='bottom')

    plt.show()

def plot_action_distribution(action_counts: Counter):
    """Plots the distribution of action types."""
    if not action_counts:
        log("No action data to plot.")
        return

    # Sort actions by count for better visualization
    sorted_actions = sorted(action_counts.items(), key=lambda item: item[1], reverse=True)
    action_types = [action.name for action, count in sorted_actions]
    counts = [count for action, count in sorted_actions]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(action_types, counts)
    plt.title('Distribution of Agent Actions During Training')
    plt.xlabel('Action Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(int(yval)), va='bottom', ha='center') # Add text labels

    plt.show()

if __name__ == "__main__":
    # Get the user's home directory
    home_dir = Path.home()

    # Construct the path to the Downloads folder
    downloads_dir = home_dir / "Downloads"
    # Define the path to the memory file
    memory_file = downloads_dir / "memory.pkl"
    
    # Load the memory
    agent_memory = load_memory(memory_file)
    
    if agent_memory:
        # Analyze win/loss outcomes, steps per episode, and action counts
        episode_outcomes, episode_steps, action_distribution = analyze_win_loss_and_steps(agent_memory)

        if episode_outcomes:
            # Calculate win rate over time (e.g., in chunks of 100 episodes)
            chunk_size = 100
            chunks, rates = calculate_win_rate_over_time(episode_outcomes, chunk_size)

            # Plot the results
            plot_win_rate(chunks, rates)

            # Plot action distribution
            plot_action_distribution(action_distribution)

            # Print overall summary
            overall_win_rate = np.mean(episode_outcomes) * 100
            log(f"\n--- Analysis Summary ---")
            log(f"Total valid episodes analyzed: {len(episode_outcomes)}")
            log(f"Overall Win Rate: {overall_win_rate:.2f}%")

            # Calculate and log step statistics
            if episode_steps:
                avg_steps = np.mean(episode_steps)
                median_steps = np.median(episode_steps)
                min_steps = np.min(episode_steps)
                max_steps = np.max(episode_steps)
                log(f"Average steps per episode: {avg_steps:.2f}")
                log(f"Median steps per episode: {median_steps}")
                log(f"Min steps per episode: {min_steps}")
                log(f"Max steps per episode: {max_steps}")
            else:
                log("No step data available for completed episodes.")

            # Log action distribution summary
            log("\n--- Action Distribution ---")
            if action_distribution:
                total_actions = sum(action_distribution.values())
                log(f"Total actions taken across all episodes: {total_actions}")
                # Sort actions by count for display
                sorted_actions = sorted(action_distribution.items(), key=lambda item: item[1], reverse=True)
                for action_type, count in sorted_actions:
                    percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                    log(f"- {action_type.name}: {count} ({percentage:.2f}%)")
            else:
                log("No action data recorded.")
            log(f"------------------------")

        else:
            log("No valid episode outcomes found to calculate statistics.")
    else:
        log("Could not load memory or memory is empty.")