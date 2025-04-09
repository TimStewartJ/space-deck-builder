# Space Deck Builder

This project is a simulator for a space-based deck building card game, following the same rules and card mechanics as Star Realms. It is designed to facilitate the development of AI players and includes a modular architecture for easy extensibility.

## Project Structure

- **src/**: Contains the main source code for the simulator.
  - **engine/**: Manages the game state, player turns, and overall game flow.
    - `game.py`: Contains the Game class.
    - `player.py`: Defines the Player class.
    - `turn.py`: Manages the Turn class.
    - `rules.py`: Contains the Rules class.
  - **cards/**: Handles card definitions and effects.
    - `card.py`: Defines the Card class.
    - `loader.py`: Loads card definitions from a CSV file.
    - `effects.py`: Defines various card effects.
    - `factions.py`: Contains faction definitions and mechanics.
  - **ai/**: Implements AI players and strategies.
    - `agent.py`: Defines the Agent class.
    - `random_agent.py`: Implements a simple random decision-making AI.
    - `strategies.py`: Contains various AI strategies.
  - **ui/**: Manages the user interface.
    - `cli.py`: Command-line interface for the game.
    - `game_state.py`: Represents the current game state for the UI.
  - **utils/**: Contains utility functions and constants.
    - `logger.py`: Functions for logging game events.
    - `constants.py`: Defines various constants used throughout the project.

- **data/**: Contains data files, including card definitions in CSV format.
  - `cards.csv`: Card definitions.

- **tests/**: Contains unit tests for various modules.
  - `test_engine.py`: Tests for the engine module.
  - `test_cards.py`: Tests for the cards module.
  - `test_ai.py`: Tests for the AI module.

- **examples/**: Provides example scripts for using the simulator.
  - `simple_game.py`: Example of setting up and running a simple game.
  - `ai_tournament.py`: Demonstrates running a tournament between AI agents.

- **requirements.txt**: Lists the dependencies required for the project.

- **pyproject.toml**: Contains project metadata and configuration for package management.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd space-deck-builder
   ```

2. Install dependencies using the uv package manager:
   ```
   uv install
   ```

3. Run the simulator:
   ```
   python src/ui/cli.py
   ```

## Usage Examples

- To run a simple game, execute:
  ```
  python examples/simple_game.py
  ```

- To run an AI tournament, execute:
  ```
  python examples/ai_tournament.py
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.