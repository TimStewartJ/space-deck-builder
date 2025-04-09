from src.ai.agent import Agent
from src.engine.actions import get_available_actions

class HumanAgent(Agent):
    def __init__(self, name="Human", cli_interface=None):
        super().__init__(name)
        self.cli_interface = cli_interface
    
    def make_decision(self, game_state):
        """Let a human player choose an action through CLI"""
        available_actions = get_available_actions(game_state, game_state.current_player)
        
        # If we have a CLI interface, use it
        if self.cli_interface:
            return self.cli_interface.get_player_action(available_actions)
        
        # Fallback to simple CLI if no interface provided
        print("\nAvailable actions:")
        for i, action in enumerate(available_actions):
            print(f"{i+1}. {action}")
        
        while True:
            try:
                choice = int(input("Enter your choice (number): ")) - 1
                if 0 <= choice < len(available_actions):
                    return available_actions[choice]
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")