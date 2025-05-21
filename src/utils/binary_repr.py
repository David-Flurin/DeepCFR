import logging as log

import torch

from .game import Game


def get_card_binaries(game: Game, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    
    board_binary = [] if len(game.board) == 0 else torch.cat([c.binary() for c in game.board])
    player_hands_binary = torch.cat([c.binary() for c in game.player_cards[p]])

    return torch.cat([player_hands_binary, board_binary])
    
def get_table_binary(game: Game) -> torch.tensor:
    """
    Convert the game state into a binary representation.
    """
    normalization = sum([p["funds"] - game.get_player_funds_diff(i) for i,p in enumerate(game.players)]) / len(game)

    if normalization == 0:
        raise ValueError(f"Funds normalization is zero. {game.players[p]['funds']}, {game.get_player_funds_diff(p)}, {normalization}")
    
    binary = torch.zeros(7)
    binary[0] = game.small_blind / normalization
    binary[1] = game.big_blind / normalization
    binary[2] = len(game.get_active_players()) / len(game) # players still in game

    betting_order = game.initiative_order
    try:
        idx_in_betting_order = betting_order.index(game.get_current_player())
    except ValueError:
        idx_in_betting_order = 0
    players_left = (len(betting_order) - idx_in_betting_order) / max(1,len(betting_order))
    binary[3] = players_left

    binary[4] = game.main_pot / normalization
    binary[5] = game.current_highest_bet / normalization
    binary[6] = game.get_player_pot_sum() / normalization

    return binary

def get_player_binary(game: Game, p: int):
    """
    Convert the game state into a binary representation.
    """
    normalization = sum([p["funds"] - game.get_player_funds_diff(i) for i,p in enumerate(game.players)]) / len(game)

    if normalization == 0:
        raise ValueError(f"Funds normalization is zero. {game.players[p]['funds']}, {game.get_player_funds_diff(p)}, {normalization}")
    
    binary = torch.zeros(4)
    binary[0] = game.players[p]["funds"] / normalization
    binary[1] = game.players[p]["pot"] / normalization
    binary[2] = 1 if game.players[p]["active"] else 0
    total_bets_this_round = game.get_player_funds_diff(p)
    is_all_in = 1 if total_bets_this_round == game.players[p]["funds"] else 0
    binary[3] = is_all_in

    return binary



