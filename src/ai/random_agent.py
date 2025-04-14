from random import choice
from src.engine.game import Game
from src.engine.actions import get_available_actions
from src.ai.agent import Agent

class RandomAgent(Agent):
    def make_decision(self, game_state: Game):
        # Randomly choose an action from the available options
        available_actions = get_available_actions(game_state, game_state.current_player)

        if len(available_actions) == 1:
            return available_actions[0]
        
        # Remove the last action from the list
        available_actions.pop()

        return choice(available_actions)
