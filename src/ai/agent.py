from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.actions import Action

class Agent:
    def __init__(self, name, cli_interface=None):
        self.name = name
        self.cli_interface = cli_interface

    def make_decision(self, game_state) -> 'Action':
        raise NotImplementedError("This method should be overridden by subclasses")