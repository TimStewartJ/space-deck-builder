from random import choice
from src.ai.agent import Agent

class RandomAgent(Agent):
    def make_decision(self, game_state):
        # Randomly choose an action from the available options
        available_actions = self.get_available_actions(game_state)
        if available_actions:
            return choice(available_actions)
        return None

    def get_available_actions(self, game_state):
        # This method should return a list of actions that the agent can take
        actions = []
        # Logic to populate actions based on the game state
        return actions