import logging
import random
from copy import copy

import numpy as np

from .constants import ACTIONS, BETTING_ROUNDS
from .deck import Deck
from .helpers import get_winners

log = logging.getLogger(__name__)

class Table:
    """Representation of a game of Texas Holdem No-Limit Poker."""

    def __init__(self,n_players: int, funds:float, small_blind: float, big_blind: float):
        """Initialize the game with players and blinds."""

        self.n_players = n_players
        self.players = [{"funds": funds, "active":True, "pot":0} for _ in range(n_players)]
        self.dealer = random.randint(0, n_players - 1)
        self.small_blind = small_blind
        self.big_blind = big_blind

    def __len__(self):
        """Return the number of players."""
        return self.n_players

    def reset_players(self):
        """Reset the players' states for a new game."""
        for player in self.players:
            player["active"] = True
            player["pot"] = 0

    def reset_players_pots(self):
        """Reset the players' pots for a new betting round."""
        for player in self.players:
            player["pot"] = 0

    def get_player_pot_sum(self):
        """Get the sum of all players' pots."""
        return sum(player["pot"] for player in self.players)
    
    def add_funds(self, player_index: int, amount: float):
        """Add funds to a player's account."""
        assert 0 <= player_index < self.n_players
        self.players[player_index]["funds"] += amount

    def move_dealer(self):
        """Move the dealer position to the next player."""
        self.dealer = (self.dealer + 1) % self.n_players

    def get_active_players(self):
        """Get a list of active players."""
        return [i for i, player in enumerate(self.players) if player["active"]]
    
    def get_next_active_player(self, player_index: int):
        """Get the next active player after a given index."""
        for i in range(1, self.n_players):
            next_player = (player_index + i) % self.n_players
            if self.players[next_player]["active"]:
                return next_player
        return None



class Game:
    def __init__(self, funds: list[float], small_blind: float, big_blind: float, player_cards=None):
        """Initialize the game with players and blinds."""

        self.n_players = len(funds)
        self.players = [{"funds": f, "active":True, "pot":0} for f in funds]
        self.dealer = random.randint(0, self.n_players - 1)
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.board = []
        self.main_pot = 0
        self.side_pots = []
        self.current_highest_bet = 0
        self.current_betting_round = BETTING_ROUNDS["pre_flop"]
        self._player_funds_diff = np.zeros((self.n_players, len(BETTING_ROUNDS)))

        self.deck = Deck()

        p_sm = (self.dealer + 1) % self.n_players
        p_bg = (self.dealer + 2) % self.n_players
        
        
        if player_cards is None:
            self.player_cards = [[] for _ in range(self.n_players)]
            self._deal_hole_cards()
            
            self.bet(p_sm, small_blind)
            self.bet(p_bg, big_blind)

        else:
            assert len(player_cards) == self.n_players
            self.player_cards = player_cards
        
        self.initiative_order = self._place_initiative_order((p_bg + 1) % self.n_players, skip_last_player=False)
        self.current_player = self._get_next_player()

    def __len__(self):
        """Return the number of players."""
        return self.n_players
    
    def __copy__(self):
        """Return a copy of the game."""
        new_game = Game([0 for _ in self.players], self.small_blind, self.big_blind, self.player_cards)
        new_game.players = [{"funds": p["funds"], "active": p["active"], "pot": p["pot"]} for p in self.players]
        new_game.dealer = self.dealer
        new_game.board = self.board.copy()
        new_game.main_pot = self.main_pot
        new_game.side_pots = self.side_pots.copy()
        new_game.current_highest_bet = self.current_highest_bet
        new_game.current_betting_round = self.current_betting_round
        new_game._player_funds_diff = np.copy(self._player_funds_diff)
        new_game.current_player = self.current_player
        new_game.initiative_order = self.initiative_order.copy()
        new_game.deck = copy(self.deck)
        log.debug(f"Copied game with funds: {[p['funds'] for p in new_game.players]} and pots: {[p['pot'] for p in new_game.players]} and main pot: {new_game.main_pot} and initiative order: {new_game.initiative_order}")
        return new_game
    

    def _place_initiative_order(self, start_player: int, skip_last_player: bool = True):
        """Get the order of players for the current betting round."""
        order = []
        for i in range(self.n_players if not skip_last_player else self.n_players - 1):
            next_player = (start_player + i) % self.n_players
            if self.players[next_player]["active"]:
                order.append(next_player)
        return order
        

    def _get_next_player(self):
        """Get the next player in the betting round."""
        try:
            return self.initiative_order.pop(0)
        except IndexError:
            return None

    def _deal_hole_cards(self):
        """Deal two hole cards to each player."""

        for i in range(self.n_players):
            cards = [self.deck.draw(), self.deck.draw()]
            self.player_cards[i] = sorted(cards, key=lambda x: (x.rank, x.suit))

    def _p_move_chips(self, player_index: int, amount: float):
        self.players[player_index]["funds"] -= amount
        self.players[player_index]["pot"] += amount
        self._player_funds_diff[player_index][self.current_betting_round] -= amount

    def reset_players(self):
        """Reset the players' states for a new game."""
        for player in self.players:
            player["active"] = True
            player["pot"] = 0

    def reset_players_pots(self):
        """Reset the players' pots for a new betting round."""
        for player in self.players:
            player["pot"] = 0

    def get_player_pot_sum(self):
        """Get the sum of all players' pots."""
        return sum(player["pot"] for player in self.players)
    
    def add_funds(self, player_index: int, amount: float):
        """Add funds to a player's account."""
        assert 0 <= player_index < self.n_players
        self.players[player_index]["funds"] += amount

    def move_dealer(self):
        """Move the dealer position to the next player."""
        self.dealer = (self.dealer + 1) % self.n_players

    def get_active_players(self):
        """Get a list of active players."""
        return [i for i, player in enumerate(self.players) if player["active"]]
    
    def get_next_active_player(self, player_index: int):
        """Get the next active player after a given index."""
        for i in range(1, self.n_players):
            next_player = (player_index + i) % self.n_players
            if self.players[next_player]["active"]:
                return next_player
        return None

    def deal_board_cards(self, amount: int):
        """Deal a specified number of community cards."""
        assert amount in [1, 3], "Only 1 or 3 cards can be dealt at once."
        
        if amount == 1:
            new_card = self.deck.draw()
            log.debug(f" Dealt card {new_card} to board")
            self.board = sorted(self.board + [new_card], key=lambda x: (x.rank, x.suit))
        elif amount == 3:
            new_cards = [self.deck.draw() for _ in range(3)]
            log.debug(f" Dealt cards {new_cards} to board")
            self.board = sorted(self.board + new_cards, key=lambda x: (x.rank, x.suit))

    def get_player_funds_diff(self, player_index: int):
        """Get the difference in funds for each player."""
        return sum(self._player_funds_diff[player_index])

    def conclude_betting_round(self):
        self.main_pot += self.get_player_pot_sum()
        self.reset_players_pots()
        self.current_highest_bet = 0

        log.debug(f"Concluded betting round {self.current_betting_round}, next player is {self.current_player}, main pot is {self.main_pot}")

        self.current_betting_round += 1
        if len(self.get_active_players()) == 1 or self.current_betting_round == BETTING_ROUNDS["showdown"]:
            self.payout()
            return

        cards_to_deal = 3 if self.current_betting_round == BETTING_ROUNDS["flop"] else 1
        self.deal_board_cards(cards_to_deal)

        self.initiative_order = self._place_initiative_order((self.dealer + 3) % self.n_players)
        self.current_player = self._get_next_player()


    def bet(self, player_index, amount):
        """Process a bet from a player."""

        assert 0 <= player_index < self.n_players
        assert amount <= self.players[player_index]["funds"] 

        self.current_highest_bet = amount + self.players[player_index]["pot"]
        self._p_move_chips(player_index, amount)

        self.initiative_order = self._place_initiative_order((player_index + 1) % self.n_players)

        log.debug(f"Player {player_index} bets {amount}, new initiative order: {self.initiative_order}, new pot is {self.players[player_index]['pot']}")

    def check_call(self, player_index):
        """Process a call from a player."""

        assert 0 <= player_index < self.n_players
        assert self.players[player_index]["active"]
        assert self.players[player_index]["pot"] <= self.current_highest_bet

        amount_to_call = self.current_highest_bet - self.players[player_index]["pot"]
        if amount_to_call > 0:
            self._p_move_chips(player_index, amount_to_call)
        
            log.debug(f"Player {player_index} calls {amount_to_call}, new pot is {self.players[player_index]['pot']}")
        else:
            log.debug(f"Player {player_index} checks")
        
        

    def fold(self, player_index):
        """Process a fold from a player."""

        assert 0 <= player_index < self.n_players
        assert self.players[player_index]["active"]

        self.players[player_index]["active"] = False

        log.debug(f"Player {player_index} folds, active players: {self.get_active_players()}")

    def get_current_player(self):
        """Get the index of the current player."""
        return self.current_player
    
    def payout (self):
        """Conclude the game and determine the winner."""

        active = self.get_active_players()
        if len(active) == 1:
            winners = active
        else:
            winners = get_winners(self.player_cards, self.board)

        for winner in winners:
            self._player_funds_diff[winner][self.current_betting_round] += self.main_pot / len(winners)

        log.debug(f"winners: {winners}, funds diff: {self._player_funds_diff}")

    def conclude_game(self):
        """Conclude the game and reset the table."""
        self.reset_players()
        for i,diff in self._player_funds_diff:
            self.add_funds(i, sum(diff))
        
        self.move_dealer()
        log.debug(f"Concluded, dealer moved to {self.dealer}")

    def get_legal_actions(self):
        """Get the legal actions for the current player."""
        actions = []
        avl_funds = self.players[self.current_player]["funds"]
        pot = self.players[self.current_player]["pot"]

        if (self.players[self.current_player]["active"] == False):
            return actions
        
        if pot < self.current_highest_bet:
            actions.append(ACTIONS["fold"])

        if pot == self.current_highest_bet or self.current_highest_bet - pot <= avl_funds:
            actions.append(ACTIONS["check/call"])
        

        min_bet = max(2 * self.current_highest_bet - pot, self.big_blind) 
        current_pot = self.get_player_pot_sum() + self.main_pot
        if min_bet > 3 * self.big_blind and avl_funds - min_bet > 0:
            actions.append(ACTIONS["minBet"])
        if min_bet <= self.big_blind * 3 and avl_funds - self.big_blind * 3 > 0:
            actions.append(ACTIONS["bet3BB"])
        if min_bet < current_pot / 2 and avl_funds - current_pot / 2 > 0:
            actions.append(ACTIONS["betHalfPot"])
        if avl_funds > self.current_highest_bet:
            actions.append(ACTIONS["betAllIn"])

        return actions
    
    def take_action(self, action):
        """Take an action for the current player."""
        assert action in self.get_legal_actions()

        log.debug(f"Player {self.current_player} takes action: {action}")
        if action == ACTIONS["fold"]:
            self.fold(self.current_player)
        elif action == ACTIONS["check/call"]:
            self.check_call(self.current_player)
        else:
            current_pot = self.get_player_pot_sum() + self.main_pot
            min_bet = max(2 * self.current_highest_bet - self.players[self.current_player]["pot"], self.big_blind) 

            if action == ACTIONS["minBet"]:
                self.bet(self.current_player, min_bet)
            elif action == ACTIONS["bet3BB"]:
                self.bet(self.current_player, 3 * self.big_blind)
            elif action == ACTIONS["betHalfPot"]:
                self.bet(self.current_player, current_pot / 2)
            elif action == ACTIONS["betAllIn"]:
                self.bet(self.current_player, self.players[self.current_player]["funds"])


        self.current_player = self._get_next_player()
        log.debug(f"Next player is {self.current_player}")
        if self.current_player is None:
            self.conclude_betting_round()

        


    



        




