import pygame
import os
from src.engine.game import Game
from src.cards.card import Card

class PygameUI:
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

    # Card dimensions
    CARD_WIDTH = 120
    CARD_HEIGHT = 180
    CARD_MARGIN = 10

    def __init__(self, width=1280, height=720):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Space Deck Builder")
        self.font = pygame.font.SysFont(None, 24)
        self.card_font = pygame.font.SysFont(None, 20)

    def draw_card(self, card: Card, x: int, y: int, highlighted=False):
        """Draw a single card at the specified position"""
        # Draw card background
        color = self.WHITE if not highlighted else self.YELLOW
        pygame.draw.rect(self.screen, color, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)

        # Draw card name
        name_text = self.card_font.render(card.name, True, self.BLACK)
        self.screen.blit(name_text, (x + 5, y + 5))

        # Draw card cost
        cost_text = self.card_font.render(f"Cost: {card.cost}", True, self.BLACK)
        self.screen.blit(cost_text, (x + 5, y + 25))

        # Draw faction if any
        if card.faction:
            faction_text = self.card_font.render(f"Faction: {card.faction}", True, self.BLACK)
            self.screen.blit(faction_text, (x + 5, y + 45))

        # Draw defense for bases
        if card.defense:
            defense_text = self.card_font.render(f"Defense: {card.defense}", True, self.RED)
            self.screen.blit(defense_text, (x + 5, y + 65))

        # Draw card effects (truncated if needed)
        y_offset = 85
        for effect in card.effects:
            if len(effect) > 25:
                effect = effect[:22] + "..."
            effect_text = self.card_font.render(effect, True, self.BLACK)
            self.screen.blit(effect_text, (x + 5, y + y_offset))
            y_offset += 20

    def draw_game_state(self, game: Game):
        """Draw the current game state"""
        self.screen.fill(self.BLACK)

        # Draw trade row
        self.draw_text("Trade Row:", 10, 10)
        for i, card in enumerate(game.trade_row):
            self.draw_card(card, 10 + i * (self.CARD_WIDTH + self.CARD_MARGIN), 40)

        # Draw current player info
        if game.current_player:
            player = game.current_player
            self.draw_text(f"Player: {player.name}", 10, 300)
            self.draw_text(f"Authority: {player.health}", 10, 330)
            self.draw_text(f"Trade: {player.trade}", 10, 360)
            self.draw_text(f"Combat: {player.combat}", 10, 390)

            # Draw player's hand
            self.draw_text("Hand:", 10, 420)
            for i, card in enumerate(player.hand):
                self.draw_card(card, 10 + i * (self.CARD_WIDTH + self.CARD_MARGIN), 450)

            # Draw player's bases
            if player.bases:
                self.draw_text("Bases:", 10, 650)
                for i, card in enumerate(player.bases):
                    self.draw_card(card, 10 + i * (self.CARD_WIDTH + self.CARD_MARGIN), 680)

        pygame.display.flip()

    def draw_text(self, text: str, x: int, y: int, color=WHITE):
        """Helper method to draw text"""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def handle_events(self):
        """Handle pygame events. Returns True if the game should continue, False if it should exit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse clicks here if needed
                pass
        return True

    def sleep(self, seconds: float):
        """Sleep for a specified number of seconds"""
        pygame.time.delay(int(seconds * 1000))

    def close(self):
        """Clean up pygame resources"""
        pygame.quit()
