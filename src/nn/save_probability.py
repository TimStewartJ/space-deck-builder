def calculate_save_probability(total_episodes: int, max_saved: int) -> float:
    """
    Calculate the percent chance to save an episode.

    Args:
        total_episodes (int): The total number of episodes.
        max_saved (int): The maximum number of episodes to save.
    
    Returns:
        float: The chance (0 to 1) to save an episode.
    """
    if total_episodes <= 0:
        raise ValueError("Total episodes must be a positive integer.")
    
    # Ensure we don't exceed 100% chance
    probability = min(max_saved / total_episodes, 1.0)
    return probability
