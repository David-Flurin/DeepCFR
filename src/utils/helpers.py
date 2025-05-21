import logging as log

import numpy as np
from treys import Evaluator

from .deck import Card


def get_winners(hands: list[list[Card]], board:list[Card]) -> np.ndarray:
    """
    Get the winner of a poker hand given the hands and board.
    """
    evaluator = Evaluator()
    scores = np.zeros(len(hands))
    # Evaluate each hand against the board

    treys_board = [c.treys() for c in board]
    for i, hand in enumerate(hands):
        log.debug(f"Evaluating hand {i}: {hand} against board: {board}")
        # Evaluate the hand against the board
        scores[i] = evaluator.evaluate(treys_board, [c.treys() for c in hand])
    
    # Get the index of the winning hand
    return np.where(scores == np.min(scores))[0]