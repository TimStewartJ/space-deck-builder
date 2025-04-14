from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.utils.logger import log

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

def analyze_win_loss(memory: List[List[Tuple]]) -> List[int]:
    """
    Analyzes the memory to determine win/loss outcomes for each episode.
    Assumes the NeuralAgent is Player 1 and a positive reward at the end signifies a win.

    Args:
        memory: A list of episodes, where each episode is a list of transitions.

    Returns:
        A list of integers representing outcomes (1 for win, 0 for loss).
    """
    outcomes = []
    for i, episode in enumerate(memory):
        if not episode:
            log(f"Warning: Episode {i} is empty.")
            continue
            
        # The last transition in the episode should have done=True
        last_state, last_action, last_reward, _, done = episode[-1]
        
        if not done:
            log(f"Warning: Episode {i} did not end with done=True. Last transition reward: {last_reward}")
            # Cannot reliably determine outcome, skip or make assumption
            continue 

        # Check the reward of the final step. Based on trainer.py, win reward is 1000, loss is -1000.
        # We assume the reward recorded is for the agent being trained.
        if abs(last_reward) != 1000:
            log(f"Warning: Episode {i} has an unexpected reward value: {last_reward}. Expected -1000 or 1000.")
            continue
        if last_reward > 0:
            outcomes.append(1)  # Win
        else:
            outcomes.append(0)  # Loss or Draw (treat as loss for win rate)
            
    log(f"Analyzed {len(outcomes)} completed episodes.")
    return outcomes

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

    plot_filename = "win_rate_over_time.png"
    plt.savefig(plot_filename)
    log(f"Win rate plot saved to {plot_filename}")
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
        # Analyze win/loss outcomes
        episode_outcomes = analyze_win_loss(agent_memory)
        
        if episode_outcomes:
            # Calculate win rate over time (e.g., in chunks of 100 episodes)
            chunk_size = 100 
            chunks, rates = calculate_win_rate_over_time(episode_outcomes, chunk_size)
            
            # Plot the results
            plot_win_rate(chunks, rates)
            
            # Print overall summary
            overall_win_rate = np.mean(episode_outcomes) * 100
            log(f"\nOverall Win Rate across {len(episode_outcomes)} episodes: {overall_win_rate:.2f}%")
        else:
            log("No valid episode outcomes found to calculate win rate.")
    else:
        log("Could not load memory or memory is empty.")