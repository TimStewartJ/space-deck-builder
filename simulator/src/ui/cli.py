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
        self.use_pygame = False
        self.pygame_ui = None
        self.available_agents = self._discover_agents()
        self.games_count = 1
        self.win_stats = {}
        # Add aggregate statistics dictionaries
        self.aggregate_stats = {
            'total_turns': [],
            'game_durations': [],
            'player_stats': {}
        }

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
    
    def _reset_aggregate_stats(self, player1_name, player2_name):
        """Reset aggregate statistics for a new set of games"""
        self.win_stats = {player1_name: 0, player2_name: 0}
        self.aggregate_stats = {
            'total_turns': [],
            'game_durations': [],
            'player_stats': {
                player1_name: {
                    'cards_played': [],
                    'cards_scrapped': [],
                    'cards_bought': [],
                    'damage_dealt': [],
                    'trade_generated': [],
                    'cards_drawn': [],
                    'bases_destroyed': [],
                    'authority_gained': []
                },
                player2_name: {
                    'cards_played': [],
                    'cards_scrapped': [],
                    'cards_bought': [],
                    'damage_dealt': [],
                    'trade_generated': [],
                    'cards_drawn': [],
                    'bases_destroyed': [],
                    'authority_gained': []
                }
            }
        }

    def _update_aggregate_stats(self):
        """Update aggregate statistics after each game"""
        if not self.game or not self.game.stats:
            return

        # Record game-level stats
        self.aggregate_stats['total_turns'].append(self.game.stats.total_turns)
        self.aggregate_stats['game_durations'].append(self.game.stats.get_game_duration())

        # Record per-player stats
        for player_name, stats in self.game.stats.player_stats.items():
            player_stats = self.aggregate_stats['player_stats'][player_name]
            player_stats['cards_played'].append(stats.cards_played)
            player_stats['cards_scrapped'].append(stats.cards_scrapped)
            player_stats['cards_bought'].append(stats.cards_bought)
            player_stats['damage_dealt'].append(stats.damage_dealt)
            player_stats['trade_generated'].append(stats.trade_generated)
            player_stats['cards_drawn'].append(stats.cards_drawn)
            player_stats['bases_destroyed'].append(stats.bases_destroyed)
            player_stats['authority_gained'].append(stats.authority_gained)
        
    def _display_aggregate_stats(self):
                        
        # Print final statistics
        print("\nOverall Results:")
        print(f"{self.aggregate_stats}")
    
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
        print("  pygame - Toggle pygame visualization")
        print("  games N - Set number of games to simulate (default: 1)")
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
            
            if command.startswith("games "):
                try:
                    num = int(command.split()[1])
                    if num > 0:
                        self.games_count = num
                        print(f"Number of games set to {num}")
                    else:
                        print("Number of games must be positive")
                except (IndexError, ValueError):
                    print("Usage: games N (where N is a positive number)")
            
            elif command in ["start", "s"]:
                agents = list(self.available_agents.items())
                
                def select_agent(player_num):
                    print(f"\nSelect agent for Player {player_num}:")
                    for i, (name, _) in enumerate(agents):
                        print(f"{i + 1}. {name}")
                    while True:
                        try:
                            choice = int(input("Enter your choice (number): ")) - 1
                            if 0 <= choice < len(agents):
                                agent_name, agent_class = agents[choice]
                                return agent_name, agent_class(name=agent_name, cli_interface=self)
                            else:
                                print("Invalid choice. Try again.")
                        except ValueError:
                            pass
                        print(f"Please enter a number between 1 and {len(agents)}")

                # Select agents for both players
                name1, agent1 = select_agent(1)
                name2, agent2 = select_agent(2)

                # add numbers to the names
                name1 += " (Player 1)"
                name2 += " (Player 2)"
                
                # Initialize win statistics
                self._reset_aggregate_stats(name1, name2)
                
                # Run multiple games
                for game_num in range(self.games_count):
                    if self.games_count > 1:
                        print(f"\nStarting game {game_num + 1} of {self.games_count}")
                    
                    cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])
                    self.game = Game(cards, verbose=self.verbose)
                    
                    # Add both players with their selected agents
                    player1 = self.game.add_player(name1)
                    player1.agent = agent1
                    
                    player2 = self.game.add_player(name2)
                    player2.agent = agent2
                    
                    # Initialize pygame UI if enabled (only for single game mode)
                    if self.use_pygame and self.games_count == 1:
                        from src.ui.pygame_ui import PygameUI
                        self.pygame_ui = PygameUI()

                    self.game.start_game()
                    if self.games_count == 1:
                        print(f"Game started: {name1} vs {name2}!")
                    
                    # Main game loop
                    running = True
                    while running and not self.game.is_game_over:
                        if self.games_count == 1:
                            self.display_game_state()
                        if self.pygame_ui:
                            running = self.pygame_ui.handle_events()
                            self.pygame_ui.draw_game_state(self.game)
                        self.game.next_turn()

                    # Record winner
                    winner = self.game.get_winner()
                    if winner == "Player 1":
                        self.win_stats[name1] += 1
                    elif winner == "Player 2":
                        self.win_stats[name2] += 1

                    # Update aggregate statistics
                    self._update_aggregate_stats()

                    # Display game statistics
                    print("\n" + "="*50)
                    print(self.game.stats.get_summary())
                    print("="*50)

                    # Clean up pygame
                    if self.pygame_ui:
                        self.pygame_ui.close()
                        self.pygame_ui = None

                # Display aggregate statistics
                self._display_aggregate_stats()
            
            elif command == "verbose":
                self.verbose = not self.verbose
                if self.game:
                    self.game.verbose = self.verbose
                print(f"Verbose mode {'enabled' if self.verbose else 'disabled'}")
            
            elif command == "pygame":
                self.use_pygame = not self.use_pygame
                print(f"Pygame UI {'enabled' if self.use_pygame else 'disabled'}")
            
            elif command in ["help", "h", "?"]:
                self.display_help()
            
            elif command in ["quit", "q", "exit"]:
                break
            
            elif command:
                print("Unknown command. Type 'help' for available commands.")

if __name__ == "__main__":
    CLI().main()