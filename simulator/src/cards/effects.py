class Effects:
    @staticmethod
    def gain_combat(player, amount):
        player.combat += amount

    @staticmethod
    def draw_card(player, number_of_cards):
        for _ in range(number_of_cards):
            if player.deck:
                card = player.deck.pop()
                player.hand.append(card)

    @staticmethod
    def gain_money(player, amount):
        player.money += amount

    @staticmethod
    def trash_card(player, card):
        if card in player.hand:
            player.hand.remove(card)
            player.trash.append(card)

    @staticmethod
    def heal(player, amount):
        player.health += amount
        player.health = min(player.health, player.max_health)  # Ensure health does not exceed max health