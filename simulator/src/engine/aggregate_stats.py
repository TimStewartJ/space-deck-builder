from dataclasses import dataclass, field
from typing import Dict, List
from .game_stats import GameStats

@dataclass
class PlayerAggregateStats:
    cards_played: List[int] = field(default_factory=list)
    cards_scrapped: List[int] = field(default_factory=list)
    cards_bought: List[int] = field(default_factory=list)
    damage_dealt: List[int] = field(default_factory=list)
    trade_generated: List[int] = field(default_factory=list)
    cards_drawn: List[int] = field(default_factory=list)
    bases_destroyed: List[int] = field(default_factory=list)
    authority_gained: List[int] = field(default_factory=list)
    scrapped_from_hand: List[int] = field(default_factory=list)
    scrapped_from_discard: List[int] = field(default_factory=list)
    scrapped_from_trade: List[int] = field(default_factory=list)

@dataclass
class AggregateStats:
    total_turns: List[int] = field(default_factory=list)
    game_durations: List[float] = field(default_factory=list)
    player_stats: Dict[str, PlayerAggregateStats] = field(default_factory=dict)
    win_stats: Dict[str, int] = field(default_factory=dict)

    def reset(self, player1_name: str, player2_name: str):
        """Reset aggregate statistics for a new set of games"""
        self.total_turns = []
        self.game_durations = []
        self.win_stats = {player1_name: 0, player2_name: 0}
        self.player_stats = {
            player1_name: PlayerAggregateStats(),
            player2_name: PlayerAggregateStats()
        }

    def update(self, game_stats: GameStats, winner: str):
        """Update aggregate statistics after each game"""
        if not game_stats:
            return

        # Record game-level stats
        self.total_turns.append(game_stats.total_turns)
        self.game_durations.append(game_stats.get_game_duration())

        # Update win stats
        if winner in self.win_stats:
            self.win_stats[winner] += 1

        # Record per-player stats
        for player_name, stats in game_stats.player_stats.items():
            if player_name not in self.player_stats:
                continue
            player_stats = self.player_stats[player_name]
            player_stats.cards_played.append(stats.cards_played)
            player_stats.cards_scrapped.append(stats.cards_scrapped)
            player_stats.cards_bought.append(stats.cards_bought)
            player_stats.damage_dealt.append(stats.damage_dealt)
            player_stats.trade_generated.append(stats.trade_generated)
            player_stats.cards_drawn.append(stats.cards_drawn)
            player_stats.bases_destroyed.append(stats.bases_destroyed)
            player_stats.authority_gained.append(stats.authority_gained)
            player_stats.scrapped_from_hand.append(stats.scrapped_from_hand)
            player_stats.scrapped_from_discard.append(stats.scrapped_from_discard)
            player_stats.scrapped_from_trade.append(stats.scrapped_from_trade)

    def get_summary(self) -> str:
        """Get a formatted summary of the aggregate statistics"""
        if not self.total_turns:
            return "No games played yet"

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        summary = [
            f"\nAggregate Statistics over {len(self.total_turns)} games:",
            f"Game Duration: min={min(self.game_durations):.1f}s, max={max(self.game_durations):.1f}s, avg={avg(self.game_durations):.1f}s",
            f"Total Turns: min={min(self.total_turns)}, max={max(self.total_turns)}, avg={avg(self.total_turns):.1f}",
            "\nWin Statistics:"
        ]

        for player_name, wins in self.win_stats.items():
            win_rate = (wins / len(self.total_turns)) * 100
            summary.append(f"  {player_name}: {wins} wins ({win_rate:.1f}%)")
            
        summary.append("\nPer Player Statistics:")
        for player_name, stats in self.player_stats.items():
            summary.extend([
                f"\n{player_name} Statistics:",
                f"  Cards Played: min={min(stats.cards_played)}, max={max(stats.cards_played)}, avg={avg(stats.cards_played):.1f}",
                f"  Cards Scrapped: min={min(stats.cards_scrapped)}, max={max(stats.cards_scrapped)}, avg={avg(stats.cards_scrapped):.1f}",
                f"  Cards Bought: min={min(stats.cards_bought)}, max={max(stats.cards_bought)}, avg={avg(stats.cards_bought):.1f}",
                f"  Damage Dealt: min={min(stats.damage_dealt)}, max={max(stats.damage_dealt)}, avg={avg(stats.damage_dealt):.1f}",
                f"  Trade Generated: min={min(stats.trade_generated)}, max={max(stats.trade_generated)}, avg={avg(stats.trade_generated):.1f}",
                f"  Cards Drawn: min={min(stats.cards_drawn)}, max={max(stats.cards_drawn)}, avg={avg(stats.cards_drawn):.1f}",
                f"  Bases Destroyed: min={min(stats.bases_destroyed)}, max={max(stats.bases_destroyed)}, avg={avg(stats.bases_destroyed):.1f}",
                f"  Authority Gained: min={min(stats.authority_gained)}, max={max(stats.authority_gained)}, avg={avg(stats.authority_gained):.1f}",
                f"  Scrapped from Hand: min={min(stats.scrapped_from_hand)}, max={max(stats.scrapped_from_hand)}, avg={avg(stats.scrapped_from_hand):.1f}",
                f"  Scrapped from Discard: min={min(stats.scrapped_from_discard)}, max={max(stats.scrapped_from_discard)}, avg={avg(stats.scrapped_from_discard):.1f}",
                f"  Scrapped from Trade Row: min={min(stats.scrapped_from_trade)}, max={max(stats.scrapped_from_trade)}, avg={avg(stats.scrapped_from_trade):.1f}"
            ])

        return "\n".join(summary)
