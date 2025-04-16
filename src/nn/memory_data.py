from dataclasses import dataclass
from typing import List, Dict
from src.nn.experience import Experience

@dataclass
class MemoryData:
    """
    Data structure for persisting training memory, including saved episodes and batch winners.
    """
    # A list of episodes, where each episode is a sequence of Experience objects from one gameplay.
    episodes: List[List[Experience]]
    # Number of episodes contained in each batch.
    batch_size: int
    # Lambda parameter (λ) used in eligibility traces for weighting future rewards.
    lambda_param: float
    # Rate at which the exploration epsilon decays over time.
    exploration_decay_rate: float
    # Records of winners for each batch of episodes, where each entry is a dictionary mapping player IDs to their respective win counts.
    batch_winners: List[Dict[str, int]]
    # History of epsilon values over training to track exploration rate progression.
    epsilon_progression: List[float]
