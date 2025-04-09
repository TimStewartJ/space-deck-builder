class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.deck = []
        self.discard_pile = []
        self.health = 50  # Default health for a player

    def draw_card(self):
        if self.deck:
            card = self.deck.pop()
            self.hand.append(card)
            return card
        return None

    def play_card(self, card):
        if card in self.hand:
            self.hand.remove(card)
            # Implement card effect logic here
            return True
        return False

    def end_turn(self):
        self.discard_pile.extend(self.hand)
        self.hand.clear()
        # Additional logic for ending the turn can be added here

    def shuffle_deck(self):
        import random
        random.shuffle(self.deck)