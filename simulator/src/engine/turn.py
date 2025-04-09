class Turn:
    def __init__(self, player):
        self.player = player
        self.actions = []

    def execute_action(self, action):
        if self.validate_action(action):
            self.actions.append(action)
            action.perform(self.player)
        else:
            raise ValueError("Invalid action")

    def resolve_turn(self):
        for action in self.actions:
            action.resolve(self.player)
        self.actions.clear()

    def validate_action(self, action):
        # Implement validation logic for actions
        return True  # Placeholder for actual validation logic