"""
deep_crf.py

Implements the Deep Counterfactual Regret Minimization (Deep CFR) algorithm
for Texas Hold'em No Limit Poker.

Author: David Niederberger
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter as performance

import numpy as np
import torch
from torch.optim import Adam

from memory.memory_manager import MemoryManager as MM
from model_manager import ModelManager
from nets.deep_crf_net import Model
from nets.losses import weighted_mse_loss
from sampler import Sampler
from training import Trainer
from utils.binary_repr import (get_card_binaries, get_player_binary,
                               get_table_binary)
from utils.constants import ACTIONS
from utils.game import Game

device = torch.device("mps" if torch.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

class DeepCRF:
    """
    Representation of the deep counterfactual regret minimization algorithm for Texas Holdem No Limit Poker.
    """
    def __init__(self, n_players: int, memory_size: int, name=None, n_iterations=10, sampler_iterations=1000, start_stack=200, min_stack=50, n_samples=1000000, n_epochs = 1000, batch_size=4000, max_workers=8):
        self.n_players = n_players
        self.memory_size = memory_size
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.name = name if name is not None else f"deep_crf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.path = Path(__file__).parent / '../runs' / self.name
        os.makedirs(self.path, exist_ok=True)

        self.model_manager = ModelManager(n_players, 
                                          lambda: Model(4, 7, 4, n_players, len(ACTIONS)), 
                                          self.path)

        self.memory_manager = MM(n_players, memory_size, self.path)

        self.sampler = Sampler(n_players, self.memory_manager, self.model_manager, sampler_iterations, start_stack, min_stack, max_workers)

    def run(self):
        """
        Run the sampler to sample infosets and train the model.
        """
        log.info("Starting Deep CRF algorithm")
        start_i = self.memory_manager.get_min_iteration() + 1
        start_player = self.memory_manager.get_min_player()
        log.info(f"Starting iteration {start_i} for player {start_player}")

        # self.train_p(0, 2)
        for i in range(start_i, self.n_iterations):
            log.info(f"Iteration {i}")
            for p in range(start_player, self.n_players):
                log.info(f"Player {p} - Sampling infosets")
                self.sampler.sample_infosets(p, i)

                log.info(f"Player {p} - Sampled {self.memory_manager.length_p(p)} infosets")
                log.info(f"Player {p} - Training model")
                self.train_p(p, i)

                self.memory_manager.dump_p(p)
                self.model_manager.save_p(p, i)

                start_player = 0

      
            self.memory_manager.dump_s()
        self.train_s(self.n_iterations-1)
        self.model_manager.save_s()


    def train_p(self,p:int, iteration: int):
        """
        Run a training loop for player p.
        """
        self.model_manager.train_p(p)
        model = self.model_manager.get_p_model(p, device)
        trainer = Trainer(model, Adam(model.parameters(), 0.001, (0.9, 0.999)), weighted_mse_loss, self.memory_manager.p_memory[p], self.n_samples, iteration)
        trainer.train(self.n_epochs, self.batch_size)

        

    def train_s(self, iteration: int):
        """
        Run a training loop for the final strategy model
        """
        self.model_manager.train_s()
        trainer = Trainer(self.model_manager.s_model, Adam(self.model_manager.s_model.parameters(), 0.001, (0.9, 0.999)), weighted_mse_loss, self.memory_manager.s_memory, self.n_samples, iteration)
        trainer.train(100, 10000)

    
    
    def binary_infoset(self, game: Game, p: int):
        """
        Convert the game state into a binary infoset.
        """
        card_binary = get_card_binaries(game, p)
        table_binary = get_table_binary(game)
        player_binary = np.concatenate([get_player_binary(game, p) for p in range(self.n_players)])

        return np.concatenate([card_binary, table_binary, player_binary])


