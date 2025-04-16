from dataclasses import dataclass
from typing import List, Dict
from src.nn.experience import Experience

@dataclass
class MemoryData:
    """
    Data structure for persisting training memory, including saved episodes and batch winners.
    """
    episodes: List[List[Experience]]
    batch_size: int
    lambda_param: float
    exploration_decay_rate: float
    batch_winners: List[Dict[str, int]]
