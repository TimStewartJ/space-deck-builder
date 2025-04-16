from pathlib import Path
import pickle
from typing import List, Tuple
from src.nn.memory_data import MemoryData

def load_memory(memory_file_path: Path):
    """Loads the agent's memory from a pickle file."""
    with open(memory_file_path, 'rb') as f:
        memory: MemoryData = pickle.load(f)
    print(f"Successfully loaded {len(memory.episodes)} episodes from {memory_file_path}.")
    return memory
