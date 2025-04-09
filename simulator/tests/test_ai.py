import unittest
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.engine.game import Game
from src.engine.player import Player

class TestAI(unittest.TestCase):

    def setUp(self):
        self.game = Game()
        self.player1 = Player("Player 1")
        self.player2 = RandomAgent("AI 1")
        self.game.add_player(self.player1)
        self.game.add_player(self.player2)

    def test_random_agent_decision(self):
        initial_state = self.game.get_state()
        decision = self.player2.make_decision(initial_state)
        self.assertIsNotNone(decision, "AI should make a decision")
        # Add more assertions based on expected decision structure

    def test_agent_can_play_card(self):
        self.player1.draw_card()
        card_to_play = self.player1.hand[0]
        self.player1.play_card(card_to_play)
        self.assertIn(card_to_play, self.player1.played_cards, "Card should be in played cards after playing")

    def test_agent_turn_execution(self):
        self.player2.take_turn(self.game)
        self.assertTrue(self.player2.has_taken_turn, "AI should have taken a turn")

if __name__ == '__main__':
    unittest.main()