def calculate_exploration_decay_rate(total_batches: int, min_exploration: float = 0.1, decaying_batches_percentage: float = 0.8, initial_rate: float = 1.0) -> float:
    """
    Calculate the per-batch exploration decay rate.
    
    The decay is computed so that, starting at an exploration rate of ~initial_rate,
    it decays to min_exploration after a specified percentage of the total batches.
    
    Args:
        total_batches (int): The total number of training batches.
        min_exploration (float): The minimum exploration rate desired at the specified percentage of batches.
        decaying_batches_percentage (float): The percentage of the total batches over which the decay occurs.
        initial_rate (float): The starting exploration rate before decay. Defaults to 1.0.
        
    Returns:
        float: The decay rate factor to multiply the exploration rate after each batch.
    """
    # Calculate the number of batches over which the decay should occur (specified percentage of total)
    decay_batches = int(total_batches * decaying_batches_percentage)

    if decay_batches == 0:
        raise ValueError("Decay batches must be greater than zero.")
    if initial_rate <= 0:
        raise ValueError("Initial rate must be greater than zero.")
    
    # Calculate the decay rate such that: (initial_rate * decay_rate^decay_batches) = min_exploration
    decay_rate = (min_exploration / initial_rate) ** (1 / decay_batches)
    return decay_rate