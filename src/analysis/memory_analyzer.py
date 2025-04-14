from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.utils.logger import log
from collections import Counter
from src.engine.actions import Action, ActionType

def analyze_win_loss_and_steps(memory: List[List[Tuple]]) -> Tuple[List[int], List[int], List[int], List[int], Counter]:
    """
    Analyzes the memory to determine win/loss outcomes, steps per episode,
    and the frequency of different action types.
    Assumes the NeuralAgent is Player 1 and a positive reward at the end signifies a win.
    Separates outcomes based on whether the agent was first player or not.

    Args:
        memory: A list of episodes, where each episode is a list of transitions.

    Returns:
        A tuple containing:
        - A list of integers representing all outcomes (1 for win, 0 for loss).
        - A list of integers representing outcomes when agent is first player.
        - A list of integers representing outcomes when agent is not first player.
        - A list of integers representing the number of steps in each valid episode.
        - A Counter object mapping ActionType to its frequency.
    """
    all_outcomes = []
    first_player_outcomes = []
    second_player_outcomes = []
    steps_per_episode = []
    action_counts = Counter()

    for i, episode in enumerate(memory):
        if not episode:
            log(f"Warning: Episode {i} is empty.")
            continue

        num_steps = 0
        is_first_player = None
        
        for transition in episode:
            state, action, reward, next_state, done = transition
            
            # Extract first player status from state (index 1 in the encoded state)
            if is_first_player is None and hasattr(state, "__getitem__"):
                try:
                    # Check if state is a tensor and get the is_first_player flag (at index 1)
                    is_first_player = bool(state[1] > 0.5)
                except (IndexError, TypeError, AttributeError):
                    # If we can't determine first player status, default to None
                    pass
                    
            if isinstance(action, Action): # Ensure action is an Action object
                action_counts[action.type] += 1
            else:
                # Handle cases where action might not be an Action object (e.g., if stored differently)
                log(f"Warning: Encountered unexpected action format in episode {i}, step {num_steps}: {action}")
            num_steps += 1
            
        # Assuming the last reward determines win/loss for the episode
        last_reward = episode[-1][2] if episode[-1][2] is not None else 0
        outcome = 1 if last_reward > 0 else 0  # 1 for Win, 0 for Loss or Draw
        
        all_outcomes.append(outcome)
        
        # Categorize outcome based on whether agent was first player or not
        if is_first_player is True:
            first_player_outcomes.append(outcome)
        elif is_first_player is False:
            second_player_outcomes.append(outcome)
        
        steps_per_episode.append(num_steps)

    log(f"Analyzed {len(all_outcomes)} completed episodes for outcomes and steps.")
    log(f"First player episodes: {len(first_player_outcomes)}, Second player episodes: {len(second_player_outcomes)}")
    log(f"Total actions analyzed: {sum(action_counts.values())}")
    
    return all_outcomes, first_player_outcomes, second_player_outcomes, steps_per_episode, action_counts

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
        
    return episode_chunks, win_rates

def plot_win_rate(all_data: Tuple[List[int], List[float]], first_player_data: Tuple[List[int], List[float]], 
              second_player_data: Tuple[List[int], List[float]]):
    """
    Plots the win rate over time, with separate lines for overall win rate, 
    win rate when the agent starts first, and win rate when the agent starts second.
    
    Args:
        all_data: Tuple containing episode counts and overall win rates
        first_player_data: Tuple containing episode counts and win rates when agent starts first
        second_player_data: Tuple containing episode counts and win rates when agent starts second
    """
    all_episodes, all_win_rates = all_data
    first_player_episodes, first_player_win_rates = first_player_data
    second_player_episodes, second_player_win_rates = second_player_data
    
    if not all_episodes:
        log("No data to plot.")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot all win rates
    plt.plot(all_episodes, all_win_rates, marker='o', label='Overall Win Rate')
    
    # Plot first player win rates if available
    if first_player_episodes and first_player_win_rates:
        plt.plot(first_player_episodes, first_player_win_rates, marker='^', 
                 label='Win Rate (Agent First)', linestyle='--')
    
    # Plot second player win rates if available
    if second_player_episodes and second_player_win_rates:
        plt.plot(second_player_episodes, second_player_win_rates, marker='s', 
                 label='Win Rate (Agent Second)', linestyle='-.')
    
    plt.title('Neural Agent Win Rate Over Training Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.ylim(0, 100) # Win rate is a percentage
    plt.xlim(left=0) # Ensure x-axis starts near 0
    plt.legend(loc='best')
    
    # Add text for final win rates
    if all_win_rates:
        final_all_win_rate = all_win_rates[-1]
        final_all_episodes = all_episodes[-1]
        plt.text(final_all_episodes, final_all_win_rate, f'{final_all_win_rate:.1f}%', 
                 ha='right', va='bottom')
        
    if first_player_win_rates:
        final_first_win_rate = first_player_win_rates[-1]
        final_first_episodes = first_player_episodes[-1]
        plt.text(final_first_episodes, final_first_win_rate, f'{final_first_win_rate:.1f}%', 
                 ha='right', va='bottom')
        
    if second_player_win_rates:
        final_second_win_rate = second_player_win_rates[-1]
        final_second_episodes = second_player_episodes[-1]
        plt.text(final_second_episodes, final_second_win_rate, f'{final_second_win_rate:.1f}%', 
                 ha='right', va='bottom')

    plt.tight_layout()
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

def analyze_and_plot_win_rates_by_starting_position(memory: List[List[Tuple]], chunk_size: int = 100):
    """
    Analyzes memory data and plots win rates based on whether the agent goes first or second.
    
    Args:
        memory: A list of episodes, where each episode is a list of transitions.
        chunk_size: The number of episodes per chunk for calculating win rate.
    """
    # Analyze memory to get outcomes separated by starting position
    all_outcomes, first_player_outcomes, second_player_outcomes, steps, action_counts = analyze_win_loss_and_steps(memory)
    
    # Calculate win rates over time for all three categories
    all_chunks, all_win_rates = calculate_win_rate_over_time(all_outcomes, chunk_size)
    log(f"Overall win rate (latest {chunk_size} episodes): {all_win_rates[-1]:.2f}% over {len(all_outcomes)} episodes")
    
    # Calculate first player win rates
    first_chunks, first_win_rates = calculate_win_rate_over_time(first_player_outcomes, chunk_size)
    if first_win_rates:
        log(f"Win rate when agent starts first (latest chunk): {first_win_rates[-1]:.2f}% over {len(first_player_outcomes)} episodes")
    else:
        log("No data available for when agent starts first")
    
    # Calculate second player win rates
    second_chunks, second_win_rates = calculate_win_rate_over_time(second_player_outcomes, chunk_size)
    if second_win_rates:
        log(f"Win rate when agent starts second (latest chunk): {second_win_rates[-1]:.2f}% over {len(second_player_outcomes)} episodes")
    else:
        log("No data available for when agent starts second")
    
    # Plot win rates
    plot_win_rate(
        (all_chunks, all_win_rates),
        (first_chunks, first_win_rates),
        (second_chunks, second_win_rates)
    )
    
    # Also plot action distribution
    plot_action_distribution(action_counts)
