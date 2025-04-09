class Agent:
    def __init__(self, name):
        self.name = name

    def make_decision(self, game_state):
        raise NotImplementedError("This method should be overridden by subclasses")