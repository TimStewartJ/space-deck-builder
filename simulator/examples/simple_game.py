from src.engine.game import Game
from src.cards.loader import load_cards

def main():
    # Load cards from CSV
    cards = load_cards('data/cards.csv')

    # Create a new game instance
    game = Game(cards)

    # Start the game
    game.start_game()

    # Example game loop
    while not game.is_over():
        game.next_turn()

    # End the game
    game.end_game()

if __name__ == "__main__":
    main()