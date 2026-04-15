"""Integer-based faction identity using bitmasks.

Each faction is a single bit, enabling fast bitwise operations for
multi-faction cards and wildcard ally checks.
"""

from enum import IntFlag


class Faction(IntFlag):
    NONE = 0
    BLOB = 1
    MACHINE_CULT = 2
    STAR_EMPIRE = 4
    TRADE_FEDERATION = 8
    ALL = BLOB | MACHINE_CULT | STAR_EMPIRE | TRADE_FEDERATION  # wildcard


# Canonical mapping from CSV/display strings to Faction bits
_NAME_TO_FACTION: dict[str, Faction] = {
    "blob": Faction.BLOB,
    "machine cult": Faction.MACHINE_CULT,
    "star empire": Faction.STAR_EMPIRE,
    "trade federation": Faction.TRADE_FEDERATION,
}

# Display names for each single-bit faction
FACTION_NAMES: dict[Faction, str] = {
    Faction.BLOB: "Blob",
    Faction.MACHINE_CULT: "Machine Cult",
    Faction.STAR_EMPIRE: "Star Empire",
    Faction.TRADE_FEDERATION: "Trade Federation",
}


def parse_faction(name: str) -> Faction:
    """Convert a faction name string to a Faction bitmask.

    Handles single factions ("Blob"), multi-faction ("Blob / Star Empire"),
    and the wildcard ("*" → ALL). Case-insensitive.
    """
    name = name.strip()
    if name == "*":
        return Faction.ALL
    if " / " in name:
        result = Faction.NONE
        for part in name.split(" / "):
            result |= parse_faction(part)
        return result
    faction = _NAME_TO_FACTION.get(name.lower())
    if faction is None:
        raise ValueError(f"Unknown faction: {name!r}")
    return faction


def faction_display(faction: Faction) -> str:
    """Human-readable display string for a faction bitmask."""
    if faction == Faction.ALL:
        return "all"
    if faction == Faction.NONE:
        return "unaligned"
    parts = [FACTION_NAMES[f] for f in Faction if f in faction and f in FACTION_NAMES]
    return " / ".join(parts)
