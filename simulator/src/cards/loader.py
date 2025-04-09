def load_trade_deck_cards(file_path, filter_names=None, filter_sets=None):
    """
    Load cards from a CSV file with optional filtering by name and set.
    
    Args:
        file_path (str): Path to the CSV file containing card data
        filter_names (list[str], optional): List of card names to include. If None, include all cards.
        filter_sets (list[str], optional): List of sets to include. If None, include all sets.
    
    Returns:
        list[Card]: List of card objects that match the filters
    """
    import csv
    from .card import Card

    cards = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip non-card rows like rules, scorecard, etc.
            if row['Type'].lower() not in ['ship', 'base']:
                continue

            # Skip non-trade deck cards
            if row['Role'].lower() != 'trade deck':
                continue

            # Apply filters
            if filter_names and row['Name'] not in filter_names:
                continue
            if filter_sets and row['Set'] not in filter_sets:
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
                effects=[effect.strip() for effect in row['Text'].split('<hr>')] if row['Text'] else [],
                card_type=card_type,
                defense=defense,
                faction=faction,
                set=row['Set']
            )
                
            # Add multiple copies based on Qty
            qty = int(row.get('Qty', 1))
            print(f"Adding {qty} copies of {card}")
            cards.extend([card] * qty)
    return cards