from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class PlayerStats:
    cards_played: int = 0
    cards_scrapped: int = 0
    cards_bought: int = 0
    damage_dealt: int = 0
    trade_generated: int = 0
    cards_drawn: int = 0
    bases_destroyed: int = 0
    authority_gained: int = 0
    scrapped_from_hand: int = 0
    scrapped_from_discard: int = 0
    scrapped_from_trade: int = 0

@dataclass
class GameStats:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    total_turns: int = 0
    winner: str = None
    player_stats: Dict[str, PlayerStats] = field(default_factory=dict)
    trade_row_refreshes: int = 0
    
    def add_player(self, player_name: str):
        """Initialize stats for a new player"""
        self.player_stats[player_name] = PlayerStats()
    
    def record_card_play(self, player_name: str):
        """Record a card being played"""
        self.player_stats[player_name].cards_played += 1
    
    def record_card_scrap(self, player_name: str, source: str = None):
        """Record a card being scrapped
        
        Args:
            player_name: The name of the player scrapping the card
            source: The source of the scrapped card ('hand', 'discard', or 'trade')
        """
        stats = self.player_stats[player_name]
        stats.cards_scrapped += 1
        if source == "hand":
            stats.scrapped_from_hand += 1
        elif source == "discard":
            stats.scrapped_from_discard += 1
        elif source == "trade":
            stats.scrapped_from_trade += 1
    
    def record_card_buy(self, player_name: str):
        """Record a card being bought"""
        self.player_stats[player_name].cards_bought += 1
    
    def record_damage(self, player_name: str, amount: int):
        """Record damage being dealt"""
        self.player_stats[player_name].damage_dealt += amount
    
    def record_trade(self, player_name: str, amount: int):
        """Record trade being generated"""
        self.player_stats[player_name].trade_generated += amount
    
    def record_card_draw(self, player_name: str, count: int = 1):
        """Record cards being drawn"""
        self.player_stats[player_name].cards_drawn += count
    
    def record_base_destroy(self, player_name: str):
        """Record a base being destroyed"""
        self.player_stats[player_name].bases_destroyed += 1
    
    def record_authority_gain(self, player_name: str, amount: int):
        """Record authority (health) being gained"""
        self.player_stats[player_name].authority_gained += amount
    
    def end_game(self, winner: str):
        """Record the game ending"""
        self.end_time = datetime.now()
        self.winner = winner
    
    def get_game_duration(self) -> float:
        """Get the game duration in seconds"""
        if not self.end_time:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
    
    def get_summary(self) -> str:
        """Get a formatted summary of the game statistics"""
        duration = self.get_game_duration()
        summary = [
            f"Game Statistics",
            f"Duration: {duration:.1f} seconds",
            f"Total Turns: {self.total_turns}",
            f"Winner: {self.winner}",
            f"Trade Row Refreshes: {self.trade_row_refreshes}\n"
        ]
        
        for player_name, stats in self.player_stats.items():
            summary.extend([
                f"{player_name} Statistics:",
                f"  Cards Played: {stats.cards_played}",
                f"  Cards Bought: {stats.cards_bought}",
                f"  Damage Dealt: {stats.damage_dealt}",
                f"  Trade Generated: {stats.trade_generated}",
                f"  Cards Drawn from card effects: {stats.cards_drawn}",
                f"  Bases Destroyed: {stats.bases_destroyed}",
                f"  Authority Gained: {stats.authority_gained}",
                f"  Total Cards Scrapped: {stats.cards_scrapped}",
                f"    From Hand: {stats.scrapped_from_hand}",
                f"    From Discard: {stats.scrapped_from_discard}",
                f"    From Trade Row: {stats.scrapped_from_trade}\n"
            ])
            
        return "\n".join(summary)
