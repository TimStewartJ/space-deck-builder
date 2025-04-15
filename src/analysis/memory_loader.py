from pathlib import Path
import pickle
from typing import List, Tuple

def load_memory(memory_file_path: Path) -> List[List[Tuple]]:
    """Loads the agent's memory from a pickle file."""
    try:
        with open(memory_file_path, 'rb') as f:
            memory = pickle.load(f)
        print(f"Successfully loaded {len(memory)} episodes from {memory_file_path}.")
        return memory
    except FileNotFoundError:
        print(f"Error: Memory file not found at {memory_file_path}")
        return []
    except Exception as e:
        print(f"Error loading memory from {memory_file_path}: {e}")
        return []
