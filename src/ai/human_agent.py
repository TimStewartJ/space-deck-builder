from src.engine.game import Game
from src.ai.agent import Agent
from src.engine.actions import get_available_actions

class HumanAgent(Agent):
    def make_decision(self, game_state: Game):
        """Let a human player choose an action through CLI"""
        available_actions = get_available_actions(game_state, game_state.current_player)
        
        # Print current player stats
        print(f"Authority: {game_state.current_player.health} Trade: {game_state.current_player.trade} Combat: {game_state.current_player.combat}")

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