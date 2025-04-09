import unittest
from src.cards.loader import load_cards
from src.cards.card import Card

class TestCardLoading(unittest.TestCase):
    def setUp(self):
        self.cards = load_cards('data/cards.csv')

    def test_card_loading(self):
        self.assertGreater(len(self.cards), 0, "No cards were loaded from the CSV file.")

    def test_card_properties(self):
        for card in self.cards:
            self.assertIsInstance(card, Card, "Loaded object is not an instance of Card.")
            self.assertTrue(hasattr(card, 'name'), "Card is missing 'name' property.")
            self.assertTrue(hasattr(card, 'cost'), "Card is missing 'cost' property.")
            self.assertTrue(hasattr(card, 'effects'), "Card is missing 'effects' property.")

if __name__ == '__main__':
    unittest.main()