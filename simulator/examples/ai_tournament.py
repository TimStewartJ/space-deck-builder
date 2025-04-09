# ai_tournament.py

import random
from src.engine.game import Game
from src.ai.random_agent import RandomAgent
from src.ai.agent import Agent

def run_tournament(num_games, agents):
    results = {agent.__class__.__name__: 0 for agent in agents}

    for _ in range(num_games):
        game = Game()
        players = [agent() for agent in agents]
        game.start_game(players)

        while not game.is_over():
            for player in players:
                game.next_turn(player)

        winner = game.get_winner()
        results[winner.__class__.__name__] += 1

    return results

if __name__ == "__main__":
    agents = [RandomAgent]  # Add more agent classes here
    num_games = 100
    results = run_tournament(num_games, agents)

    for agent_name, wins in results.items():
        print(f"{agent_name}: {wins} wins")