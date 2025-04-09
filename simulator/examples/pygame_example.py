from src.engine.game import Game
from src.cards.loader import load_trade_deck_cards
from src.ui.pygame_ui import PygameUI
from src.ai.random_agent import RandomAgent

def main():
    # Load cards from CSV
    cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])

    # Create a new game instance
    game = Game(cards, verbose=True)
    
    # Add two AI players
    player1 = game.add_player()
    player1.agent = RandomAgent(name="Player 1")
    
    player2 = game.add_player()
    player2.agent = RandomAgent(name="Player 2")

    # Create UI
    ui = PygameUI()

    # Start the game
    game.start_game()

    # Game loop
    running = True
    while running and not game.is_game_over:
        # Handle UI events
        running = ui.handle_events()
        
        # Draw current game state
        ui.draw_game_state(game)

        # sleep for 1 second to simulate a turn
        ui.sleep(1)
        
        # Process game turn
        game.next_turn()

    # Clean up
    ui.close()

if __name__ == "__main__":
    main()
