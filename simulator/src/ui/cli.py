from src.engine.game import Game
from src.cards.loader import load_trade_deck_cards
import os
import importlib
import inspect
from src.ai.agent import Agent

class CLI:
    def __init__(self):
        self.game = None
        self.verbose = True
        self.available_agents = self._discover_agents()
    
    def _discover_agents(self):
        """Find all available agent classes in the ai directory"""
        agents = {}
        ai_dir = os.path.join(os.path.dirname(__file__), '..', 'ai')
        
        # List all python files in the ai directory
        for file in os.listdir(ai_dir):
            if file.endswith('.py') and not file.startswith('__'):
                module_name = file[:-3]  # Remove .py extension
                try:
                    # Import the module
                    module = importlib.import_module(f'src.ai.{module_name}')
                    
                    # Find all classes that inherit from Agent
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Agent) and 
                            obj != Agent):
                            agents[name] = obj
                except Exception as e:
                    print(f"Warning: Could not load agent from {file}: {e}")
        
        return agents
    
    def list_agents(self):
        """Display list of available agents"""
        print("\nAvailable Agents:")
        for name, agent_class in self.available_agents.items():
            print(f"  {name}")
            
    def display_welcome(self):
        print("Welcome to the Space Deck Builder!")
        print("Type 'help' for a list of commands.")
    
    def display_help(self):
        print("Available commands:")
        print("  start (or s) - Start a new game and select agents")
        print("  agents - List available AI agents")
        print("  verbose - Toggle verbose mode")
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
            
            if command in ["start", "s"]:
                agents = list(self.available_agents.items())
                
                def select_agent(player_num):
                    print(f"\nChoose agent for Player {player_num}:")
                    for i, (name, _) in enumerate(agents):
                        print(f"{i+1}. {name}")
                    
                    while True:
                        try:
                            choice = int(input("Enter your choice (number): ")) - 1
                            if 0 <= choice < len(agents):
                                agent_name, agent_class = agents[choice]
                                return agent_name, agent_class(name=agent_name, cli_interface=self)
                            else:
                                print("Invalid choice. Try again.")
                        except ValueError:
                            print("Please enter a number.")
                
                # Select agents for both players
                name1, agent1 = select_agent(1)
                name2, agent2 = select_agent(2)
                
                cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])
                self.game = Game(cards, verbose=self.verbose)
                
                # Add both players with their selected agents
                player1 = self.game.add_player()
                player1.agent = agent1
                
                player2 = self.game.add_player()
                player2.agent = agent2
                
                self.game.start_game()
                print(f"Game started: {name1} vs {name2}!")
                
                # Main game loop
                while not self.game.is_game_over:
                    self.display_game_state()
                    self.game.next_turn()
            
            elif command == "verbose":
                self.verbose = not self.verbose
                if self.game:
                    self.game.verbose = self.verbose
                print(f"Verbose mode {'enabled' if self.verbose else 'disabled'}")
            
            elif command == "help":
                self.display_help()
                
            elif command == "agents":
                self.list_agents()
                
            elif command == "exit":
                print("Exiting the game.")
                break
                
            else:
                print("Unknown command. Type 'help' for a list of commands.")

if __name__ == "__main__":
    CLI().main()