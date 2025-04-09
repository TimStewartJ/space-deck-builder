def load_cards(file_path):
    import csv
    from .card import Card

    cards = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip non-card rows like rules, scorecard, etc.
            if row['Type'].lower() not in ['ship', 'base']:
                continue
                
            # Parse defense value for bases
            defense = None
            if row['Defense']:
                # Extract just the number if there's "Outpost" text
                defense_str = row['Defense'].split()[0]
                defense = int(defense_str) if defense_str.isdigit() else None
            
            # Determine card type (ship, base, outpost)
            card_type = row['Type'].lower()
            if row['Defense'] and 'outpost' in row['Defense'].lower():
                card_type = 'outpost'
            
            # Handle cost - some cards like starting deck cards have no cost
            try:
                cost = int(row['Cost']) if row['Cost'] else 0
            except (ValueError, TypeError):
                cost = 0

            # Parse faction - handle multi-faction cards
            faction = None
            if row['Faction'] and row['Faction'].lower() != 'unaligned':
                faction = row['Faction'].split(' / ')  # Handle cards with multiple factions
                if len(faction) == 1:
                    faction = faction[0]  # Single faction as string

            card = Card(
                name=row['Name'],
                cost=cost,
                effects=row['Text'].split('<hr>') if row['Text'] else [],  # Effects are separated by <hr>
                card_type=card_type,
                defense=defense,
                faction=faction
            )
            # Add multiple copies based on Qty
            qty = int(row.get('Qty', 1))
            cards.extend([card] * qty)
    return cards