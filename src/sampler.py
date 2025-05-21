import logging
import random
from copy import copy
from time import perf_counter as performance

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from memory.memory_manager import MemoryManager as MM
from model_manager import ModelManager
from nets.deep_crf_net import Model
from utils.constants import ACTIONS, BETTING_ROUNDS
from utils.game import Game

log = logging.getLogger(__name__)

class Sampler:

    def __init__(self, n_players: int, memory_manager: MM, model_manager: ModelManager, sampler_iterations: int = 1000, start_stack: int = 200, min_stack: int = 50):
        
        self.memory_manager = memory_manager
        self.model_manager = model_manager
        self.n_players = n_players
        self.sampler_iterations = sampler_iterations
        self.start_stack = start_stack
        self.min_stack = min_stack

    def sample_infosets(self, current_p:int, iteration: int):
        """
        Play games of poker and traverse the game tree to sample infosets.
        """
        self.model_manager.eval()

        if self.model_manager.get_min_iteration() != self.memory_manager.get_min_iteration():
            log.warning(f"Model and memory manager iterations are not equal {self.model_manager.get_min_iteration()} and {self.memory_manager.get_min_iteration()}. Better start training from scratch.")

        def map_traverse():
            remaining = self.n_players * (self.start_stack - self.min_stack)

            stacks = np.full(self.n_players, self.min_stack, dtype=int)
            for i in range(self.n_players -1):
                additional = random.randint(0, remaining)
                stacks[i] += additional
                remaining -= additional
            stacks[-1] += remaining
            game = Game(stacks, 1, 2)
            self.traverse(current_p, game, iteration)

            
        # with tqdm(total=SAMPLER_ITERATIONS) as pbar:
        #     with ThreadPoolExecutor() as executor:
        #         futures = [executor.submit(map_traverse, p) for _ in range(SAMPLER_ITERATIONS)]
        #         for future in as_completed(futures):
        #             future.result()
        #             pbar.update(1)
        for _ in tqdm(range(self.sampler_iterations)):
            map_traverse()

    def traverse(self, p: int,  game: Game, iteration: int, depth: int = 0):
        """
        Traverse the history of the game to find the appropriate node for a given player.
        """

        log.debug(f"CFR:  Number of active players: {len(game.get_active_players())}") 

        curr_player = game.get_current_player()
 
        if (len(game.get_active_players()) == 1 or game.current_betting_round == BETTING_ROUNDS["showdown"]):
            # Terminal node
            funds_diff = game.get_player_funds_diff(p)
            log.debug(f"Hit terminal node {funds_diff}")
            return funds_diff / (game.big_blind * 10)
        
        infoset = Model.game_to_infoset(game, curr_player)
        with torch.no_grad():
            advantages: torch.Tensor = self.model_manager.get_p_model(curr_player)(infoset[0].unsqueeze(0), infoset[1].unsqueeze(0)).squeeze(0)
        log.debug(f"CFR:  Player: {curr_player}, Infoset: {infoset}, Advantages: {advantages}")
        legal_actions = game.get_legal_actions()

        # strategy = F.softmax(advantages * game.big_blind * 100, dim=0)
        strategy = F.softmax(advantages, dim=0)

        if curr_player == p:
            
            action_vals = torch.zeros(len(ACTIONS))
            log.debug(f"CFR:  Player: {p}, Legal actions: {legal_actions}")
            for i, action in enumerate(ACTIONS.values()):
                if action not in legal_actions:
                    continue 

                game_copy = copy(game)
                game_copy.take_action(action)

                action_vals[i] = self.traverse(p, game_copy, iteration, depth + 1)

            # Compute regret matching
            expected_value = torch.dot(strategy, action_vals)
            instant_advantages = action_vals - expected_value

            self.memory_manager.store_p(curr_player, infoset, instant_advantages, iteration)
            return expected_value
        
        self.memory_manager.store_s(infoset, strategy, iteration)
                
        # Sample action from strategy
        legal_action_advantages = torch.index_select(advantages, 0, torch.tensor(legal_actions, dtype=torch.int))
        # legal_action_probs = F.softmax(legal_action_advantages * game.big_blind * 100, dim=0)
        legal_action_probs = F.softmax(legal_action_advantages, dim=0)

        log.debug(f"CFR:  Player: {curr_player}, Legal actions: {legal_actions}, Advantages: {legal_action_advantages}, Probabilities: {legal_action_probs}")
        idx = torch.multinomial(legal_action_probs, 1).item()
        action = legal_actions[idx]
        log.debug(f"CFR:  Player: {curr_player} chose Action: {action}")

        game_copy = copy(game)
        game_copy.take_action(action)
        log.debug(f"CFR: Took action, new active players: {game_copy.get_active_players()}")

        return_val = self.traverse(p, game_copy, iteration, depth + 1)
        return return_val