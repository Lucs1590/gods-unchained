import pandas as pd
from app import set_strategy

def test_set_strategy():
    # Create a sample dataframe
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Card 1', 'Card 2', 'Card 3', 'Card 4', 'Card 5'],
        'mana': [3, 6, 2, 7, 4],
        'attack': [2, 4, 1, 5, 3],
        'health': [3, 5, 2, 6, 4],
        'type': ['Minion', 'Spell', 'Weapon', 'Minion', 'Spell'],
        'god': ['God 1', 'God 2', 'God 3', 'God 4', 'God 5']
    }
    df = pd.DataFrame(data)

    # Call the set_strategy function
    result = set_strategy(df)

    # Check if the strategy column is updated correctly
    assert result.loc[result['mana'] <= 5, 'strategy'].all() == 'early'
    assert result.loc[result['mana'] > 5, 'strategy'].all() == 'late'