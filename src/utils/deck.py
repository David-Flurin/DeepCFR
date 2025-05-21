import logging as log

import numpy as np
from treys import Card as TreysCard


class Deck:
    """Representation of a standard 52-card deck."""

    def __init__(self):
        """Initialize the deck with 52 cards."""
        self.cards = [Card(rank, suit) for rank in "23456789TJQKA" for suit in "HDSC"]
        self.shuffle()

    def __copy__(self):
        """Return a shallow copy of the deck."""
        new_deck = Deck()
        new_deck.cards = self.cards.copy()
        return new_deck

    def shuffle(self):
        """Shuffle the deck."""
        import random
        random.shuffle(self.cards)

    def draw(self):
        """Draw a card from the top of the deck."""
        return self.cards.pop() if self.cards else None
    
class Card:
    """Representation of a single card."""

    def __init__(self, rank: str, suit: str):
        """Initialize the card with a rank and suit."""
        self.rank = rank
        self.suit = suit

    def __str__(self):
        """Return a string representation of the card."""
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        """Return a string representation of the card."""
        return self.__str__()
    
    def __int__(self):
        """Return the card as an integer."""
        rank = "23456789TJQKA".index(self.rank)
        suit = "HDSC".index(self.suit)
        return rank * 4 + suit
    
    def treys(self):
        """Return the card in treys format."""
        return TreysCard.new(self.rank + self.suit.lower())
    
    def binary(self):
        """Return the card in binary format."""
        n = 17 # 13 ranks, 4 suits
        repr = np.zeros(n)
        rank = "23456789TJQKA".index(self.rank)
        suit = "HDSC".index(self.suit)
        repr[rank] = 1
        repr[suit + 13] = 1
        return repr

    


    
