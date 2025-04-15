from datetime import datetime
from src.engine.game import Game
from src.nn.state_encoder import encode_state

def worker_run_episode(episode, cards, card_names, first_agent, second_agent, first_agent_name, second_agent_name):
    """
    Runs one game episode in parallel.

    Parameters:
        episode (int): The current episode number.
        cards (list): List of cards to be used in the game.
        card_names (list): The names of the cards.
        first_agent (NeuralAgent): The agent being trained.
        second_agent (Agent): The opponent agent.
        first_agent_name (str): The name of the first agent.
        second_agent_name (str): The name of the second agent.

    Returns:
        tuple: (experiences, game_stats, winner)
            experiences (dict): Contains lists for states, actions, rewards, next_states, and dones.
            game_stats: The game statistics object.
            winner: Name of the winning agent.
    """
    # Setup a new game instance
    game = Game(cards)
    player1_name = first_agent_name
    player2_name = second_agent_name
    
    # Add players
    player1 = game.add_player(player1_name)
    player1.agent = first_agent
    player2 = game.add_player(player2_name)
    player2.agent = second_agent

    game.start_game()
    
    # Initialize experience buffers
    experiences = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": []
    }
    
    while not game.is_game_over:
        current_player = game.current_player
        # Determine if the current player is the training agent
        is_training = current_player.name == player1_name
        state = encode_state(game, is_current_player_training=is_training, cards=card_names)
        
        # Get the next action from the game (which may call the agent's decision methods)
        action = game.next_step()
        
        if is_training:
            # Recalculate current player
            current_player = game.current_player
            is_training = current_player.name == player1_name

            # Reward based on game outcome
            reward = 0.0
            if game.is_game_over:
                reward = 1.0 if game.get_winner() == player1_name else -1.0
            next_state = encode_state(game, is_current_player_training=is_training, cards=card_names)
            done = game.is_game_over

            experiences["states"].append(state)
            experiences["actions"].append(action)
            experiences["rewards"].append(reward)
            experiences["next_states"].append(next_state)
            experiences["dones"].append(done)
    winner = game.get_winner()
    return experiences, game.stats, winner