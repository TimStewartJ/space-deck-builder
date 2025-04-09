import unittest
from src.engine.game import Game
from src.engine.player import Player
from src.engine.turn import Turn
from src.engine.rules import Rules

class TestGameEngine(unittest.TestCase):

    def setUp(self):
        self.game = Game()
        self.player1 = Player("Player 1")
        self.player2 = Player("Player 2")
        self.game.add_player(self.player1)
        self.game.add_player(self.player2)

    def test_start_game(self):
        self.game.start_game()
        self.assertTrue(self.game.is_running)
        self.assertEqual(len(self.game.players), 2)

    def test_next_turn(self):
        self.game.start_game()
        current_player = self.game.current_player
        self.game.next_turn()
        self.assertNotEqual(current_player, self.game.current_player)

    def test_end_game(self):
        self.game.start_game()
        self.game.end_game()
        self.assertFalse(self.game.is_running)

    def test_player_actions(self):
        self.player1.draw_card()
        self.assertEqual(len(self.player1.hand), 1)
        self.player1.play_card(self.player1.hand[0])
        self.assertEqual(len(self.player1.hand), 0)

if __name__ == '__main__':
    unittest.main()