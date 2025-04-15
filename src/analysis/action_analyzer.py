from collections import Counter
from typing import List, Tuple
from src.engine.actions import Action

def analyze_actions(agent_memory: List[List[Tuple]]):
    """Analyzes actions in the agent's memory."""
    action_type_counts = Counter()
    card_action_counts = Counter()
    card_name_to_obj = {}

    total_actions_processed = 0
    invalid_action_formats = 0

    if agent_memory:
        for episode in agent_memory:
            for transition in episode:
                if len(transition) != 5:
                    continue

                state, action, reward, next_state, done = transition
                total_actions_processed += 1

                if isinstance(action, Action):
                    action_type_counts[action.type] += 1

                    if action.card is not None:
                        card_action_counts[(action.type, action.card.name)] += 1
                        card_name_to_obj[action.card.name] = action.card
                else:
                    action_type_counts["Unknown/Invalid Action Format"] += 1
                    invalid_action_formats += 1

    return action_type_counts, card_action_counts, card_name_to_obj, total_actions_processed, invalid_action_formats
