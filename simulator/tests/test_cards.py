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
            self.assertTrue(hasattr(card, 'card_type'), "Card is missing 'card_type' property.")
            self.assertTrue(hasattr(card, 'defense'), "Card is missing 'defense' property.")
    
    def test_card_types(self):
        ships = [card for card in self.cards if card.card_type == "ship"]
        bases = [card for card in self.cards if card.card_type == "base"]
        outposts = [card for card in self.cards if card.card_type == "outpost"]
        
        # Make sure we have all card types represented
        self.assertTrue(len(ships) > 0, "No ship cards found")
        self.assertTrue(len(bases) > 0, "No base cards found")
        self.assertTrue(len(outposts) > 0, "No outpost cards found")
        
        # Test that bases and outposts have defense values
        for base in bases + outposts:
            self.assertIsNotNone(base.defense, f"{base.name} is a base/outpost but has no defense value")
            
        # Test that ships don't have defense values
        for ship in ships:
            self.assertIsNone(ship.defense, f"{ship.name} is a ship but has a defense value")

if __name__ == '__main__':
    unittest.main()