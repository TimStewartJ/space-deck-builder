class Agent:
    def __init__(self, name, cli_interface=None):
        self.name = name
        self.cli_interface = cli_interface

    def make_decision(self, game_state):
        raise NotImplementedError("This method should be overridden by subclasses")