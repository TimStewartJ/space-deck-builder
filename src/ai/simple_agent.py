from random import choice
from src.engine.game import Game
from src.engine.actions import get_available_actions
from src.ai.agent import Agent

class SimpleAgent(Agent):
    """Takes the first available action.

    It plays its hand, buys, attacks, then uses abilities in the order they
    are listed, so it takes the first option of every OR ability and the
    first choice of every pending decision, never SKIP.
    """

    def make_decision(self, game_state: Game):
        # Choose the first available action from the list of available actions
        available_actions = get_available_actions(game_state, game_state.current_player)
        if available_actions and len(available_actions) > 0:
            return available_actions[0]
        return None
    