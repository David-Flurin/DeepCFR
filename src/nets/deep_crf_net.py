"""
Implementation of the Deep CRF model as suggested in Brown et al. (2019). 
"""

import torch
import torch.nn.functional as F
from torch import cat, nn, tensor

from utils.binary_repr import get_player_binary, get_table_binary
from utils.game import Game


class CardEmbedding(nn.Module):

    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, input):
        B, numcards = input.shape
        x = input.reshape(B * numcards)
        valid = x.ge(0).float() # −1 means ’nocard’
        x = x.clamp(min = 0)
        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1) #zero out ’nocard’ embeddings

        #sum across the cards in the hole/board
        return embs.view(B, numcards, -1).sum(1)
                         
class Model(nn.Module):
    def __init__(self, n_cardtypes, n_tablefeats, n_playerfeats, n_players, n_actions, dim = 256):
        super(Model, self).__init__()

        self.card_embeddings = nn.ModuleList(
        [CardEmbedding(dim) for _ in range(n_cardtypes)])

        self.card1 = nn.Linear(dim * n_cardtypes, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        self.bet1 = nn.Linear(n_tablefeats + (n_playerfeats * n_players), dim)
        self.bet2 = nn.Linear(dim, dim)

        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        self.actionhead = nn.Linear(dim, n_actions)

    def forward(self, cards, betfeats) -> torch.Tensor:
        """
        cards:((Nx2), (Nx3)[, (Nx1), (Nx1)])#(hole, board, [turn, river])
        bets:Nxnbetfeats
        """
        device = next(self.parameters()).device  # Get the model's device

        #1.cardbranch
        #embed hole, flop, and optionally turn and river
        card_groups = [cards[:, 0:2].to(device), cards[:, 2:5].to(device), cards[:, 5:6].to(device), cards[:, 6:7].to(device)]
        card_embs = [embedding(card_group) for embedding, card_group in zip(self.card_embeddings, card_groups)]
        card_embs = cat(card_embs, dim = 1)

        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))
                            
        #2.bet and player branch
        
        y = F.relu(self.bet1(betfeats.to(device)))
        y = F.relu(self.bet2(y) + y)

        #3.combinedtrunk
        z = cat([x, y], dim = 1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        z = F.normalize(z)

        return self.actionhead(z)
    
    @staticmethod
    def game_to_infoset(game: Game, p):
        """
        Convert a binary infoset into the format expected by the model.
        Optimized for better performance.
        """
        # Combine player cards and board cards
        cards = torch.tensor(
            [int(c) for c in game.player_cards[p]] + 
            [int(c) for c in game.board] + 
            [-1] * (5 - len(game.board)), 
            dtype=torch.int
        )

        # Compute player states and concatenate them
        player_states = [get_player_binary(game, i) for i in range(game.n_players)]
        player_state = torch.cat(player_states)

        # Combine table state and player state
        table_state = get_table_binary(game)
        return cards, torch.cat((table_state, player_state))

