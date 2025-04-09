import sys
from src.engine.game import Game
from src.ui.game_state import GameState
from src.cards.loader import load_cards

def display_welcome():
    print("Welcome to the Space Deck Builder!")
    print("Type 'help' for a list of commands.")

def display_help():
    print("Available commands:")
    print("  start - Start a new game")
    print("  exit - Exit the game")

def main():
    display_welcome()
    game = None

    while True:
        command = input("> ").strip().lower()

        if command == "start":
            cards = load_cards('data/cards.csv')
            game = Game(cards)
            game.start_game()
            print("Game started!")
            # Additional game loop logic would go here

        elif command == "help":
            display_help()

        elif command == "exit":
            print("Exiting the game.")
            sys.exit()

        else:
            print("Unknown command. Type 'help' for a list of commands.")

if __name__ == "__main__":
    main()