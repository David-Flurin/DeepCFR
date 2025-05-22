import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from multiprocessing import Manager, Process

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

    def __init__(self, n_players: int, memory_manager: MM, model_manager: ModelManager, sampler_iterations: int = 1000, start_stack: int = 200, min_stack: int = 50, max_workers: int = 8):
        
        self.memory_manager = memory_manager
        self.model_manager = model_manager
        self.n_players = n_players
        self.sampler_iterations = sampler_iterations
        self.start_stack = start_stack
        self.min_stack = min_stack
        self.max_workers = max_workers

    def sample_infosets(self, current_p:int, iteration: int):
        """
        Play games of poker and traverse the game tree to sample infosets.
        """

        if self.model_manager.get_min_iteration() != self.memory_manager.get_min_iteration():
            log.warning(f"Model and memory manager iterations are not equal {self.model_manager.get_min_iteration()} and {self.memory_manager.get_min_iteration()}. Better start training from scratch.")

        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        n_samples_per_worker = self.sampler_iterations // self.max_workers
        rest = self.sampler_iterations % self.max_workers
        inputs = [
            (self.n_players, current_p, iteration, n_samples_per_worker + (rest if i == self.max_workers - 1 else 0),
                 self.model_manager.get_p_model(current_p, 'cpu') , self.start_stack, self.min_stack, counter, lock)
            for i in range(self.max_workers)
        ]

        advantage_results = []
        strategy_results = []
        progress_process = Process(target=progress_monitor, args=(counter, self.sampler_iterations, lock))
        progress_process.start()

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, *input) for input in inputs]
            for f in futures:
                advantage_result, strategy_result = f.result()
                advantage_results += advantage_result
                strategy_results += strategy_result

        progress_process.join()

        for r in advantage_results:
            self.memory_manager.store_p(r[0], r[1], r[2], r[3])
        for r in strategy_results:
            self.memory_manager.store_s(r[0], r[1], r[2])
        
        
                
def progress_monitor(counter, total_tasks, lock):
    with tqdm(total=total_tasks) as pbar:
        previous = 0
        while True:
            with lock:
                current = counter.value
            pbar.update(current - previous)
            previous = current
            if current >= total_tasks:
                break
            time.sleep(0.1)

def worker(n_players:int, current_p:int, iteration: int, n_samples: int, model: torch.nn.Module, start_stack: int, min_stack: int, counter, lock):
    """
    Worker for multiprocessing.
    """
    model.eval()

    def map_traverse():
        remaining = n_players * (start_stack - min_stack)

        stacks = np.full(n_players, min_stack, dtype=int)
        for i in range(n_players -1):
            additional = random.randint(0, remaining)
            stacks[i] += additional
            remaining -= additional
        stacks[-1] += remaining
        game = Game(stacks, 1, 2)
        results = traverse(current_p, game, iteration, model)
        with lock:
            counter.value += 1
        return results

    advantage_results = []
    strategy_results = []
    for _ in range(n_samples):
        _, advantage_result, strategy_result = map_traverse()
        advantage_results += advantage_result
        strategy_results += strategy_result
    return advantage_results, strategy_results


def traverse(p: int,  game: Game, iteration: int, model:torch.nn.Module, depth: int = 0):
    """
    Traverse the history of the game to find the appropriate node for a given player.
    """

    log.debug(f"CFR:  Number of active players: {len(game.get_active_players())}") 

    curr_player = game.get_current_player()

    if (len(game.get_active_players()) == 1 or game.current_betting_round == BETTING_ROUNDS["showdown"]):
        # Terminal node
        funds_diff = game.get_player_funds_diff(p)
        log.debug(f"Hit terminal node {funds_diff}")
        return funds_diff / (game.big_blind * 10), [], []
    
    infoset = Model.game_to_infoset(game, curr_player)
    with torch.no_grad():
        advantages: torch.Tensor = model(infoset[0].unsqueeze(0), infoset[1].unsqueeze(0)).squeeze(0)
    log.debug(f"CFR:  Player: {curr_player}, Infoset: {infoset}, Advantages: {advantages}")
    legal_actions = game.get_legal_actions()

    # strategy = F.softmax(advantages * game.big_blind * 100, dim=0)
    strategy = F.softmax(advantages, dim=0)

    if curr_player == p:
        
        action_vals = torch.zeros(len(ACTIONS))
        advantage_results = []
        log.debug(f"CFR:  Player: {p}, Legal actions: {legal_actions}")
        for i, action in enumerate(ACTIONS.values()):
            if action not in legal_actions:
                continue 

            game_copy = copy(game)
            game_copy.take_action(action)

            action_vals[i], advantage_result, strategy_result  = traverse(p, game_copy, iteration, model, depth + 1)
            advantage_results += advantage_result

        # Compute regret matching
        expected_value = torch.dot(strategy, action_vals)
        instant_advantages = action_vals - expected_value

        advantage_results.append((curr_player, infoset, instant_advantages, iteration))
        return expected_value, advantage_results, strategy_result
    
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

    return_val, advantage_results, strategy_result = traverse(p, game_copy, iteration, model, depth + 1)
    strategy_result.append((infoset, strategy, iteration))
    return return_val, advantage_results, strategy_result