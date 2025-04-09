from src.engine.game import Game
from src.cards.loader import load_trade_deck_cards
from src.ai.human_agent import HumanAgent
from src.ai.random_agent import RandomAgent

class CLI:
    def __init__(self):
        self.game = None
    
    def display_welcome(self):
        print("Welcome to the Space Deck Builder!")
        print("Type 'help' for a list of commands.")
    
    def display_help(self):
        print("Available commands:")
        print("  start - Start a new game")
        print("  start ai - Start a game against AI")
        print("  exit - Exit the game")
    
    def get_player_action(self, available_actions):
        """Display available actions and get player choice"""
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
    
    def display_game_state(self):
        if not self.game:
            return
            
        player = self.game.current_player
        print("\n" + "="*50)
        print(f"Current player: {player.name}")
        print(f"Authority: {player.health}")
        print(f"Trade: {player.trade}, Combat: {player.combat}")
        
        print("\nHand:")
        for card in player.hand:
            print(f"  {card}")
            
        print("\nBases in play:")
        for card in player.bases:
            print(f"  {card}")
            
        print("\nTrade Row:")
        for card in self.game.trade_row:
            print(f"  {card}")
        print("="*50)
    
    def main(self):
        self.display_welcome()
        
        while True:
            command = input("> ").strip().lower()
            
            if command == "start":
                cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])
                self.game = Game(cards)
                
                # Add human player
                player1 = self.game.add_player()
                player1.agent = HumanAgent(name="Human", cli_interface=self)
                
                # Add human player 2
                player2 = self.game.add_player()
                player2.agent = HumanAgent(name="Human 2", cli_interface=self)
                
                self.game.start_game()
                print("Game started!")
                
                # Main game loop
                while not self.game.is_game_over:
                    self.display_game_state()
                    self.game.next_turn()
            
            elif command == "start ai":
                cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])
                self.game = Game(cards)
                
                # Add human player
                player1 = self.game.add_player()
                player1.agent = HumanAgent(name="Human", cli_interface=self)
                
                # Add AI player
                player2 = self.game.add_player()
                player2.agent = RandomAgent(name="AI")
                
                self.game.start_game()
                print("Game started against AI!")
                
                # Main game loop
                while not self.game.is_game_over:
                    self.display_game_state()
                    self.game.next_turn()
            
            elif command == "help":
                self.display_help()
                
            elif command == "exit":
                print("Exiting the game.")
                break
                
            else:
                print("Unknown command. Type 'help' for a list of commands.")

if __name__ == "__main__":
    CLI().main()