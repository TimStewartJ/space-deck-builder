def load_cards(file_path):
    import csv
    from .card import Card

    cards = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            card = Card(
                name=row['name'],
                cost=int(row['cost']),
                effects=row['effects'].split(';')  # Assuming effects are semicolon-separated
            )
            cards.append(card)
    return cards