import logging
import os
from typing import Callable

import torch

device = torch.device("mps" if torch.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

class ModelManager:

    def __init__(self, n_players: int, create_model: Callable[[], torch.nn.Module], save_path: str):
        self.p_models = {}
        self.s_model = create_model().to(device)
        self.create_model = create_model
        self.n_players = n_players
        self.path = save_path

        self.max_i_player = {}

    def find_latest_model_iteration(self) -> int:
        """
        Find the latest available model iteration for a given player.
        """
        model_files = [f for f in os.listdir(self.path) if f.startswith(f"model_p{p}_i") and f.endswith(".pt")]
        if not model_files:
            return 0
        max_i = max(int(f.split('_i')[-1].split('.pt')[0]) for f in model_files)
        return max_i
    
    def get_p_model(self, player:int, device: str = "cpu") -> torch.nn.Module: 
        """
        Get the model for a given player.
        """
        model = self.create_model().to(device)
        log.debug(f"Creating model for player {player}")
        max_i = self.find_latest_model_iteration()
        self.max_i_player[player] = max_i
        if max_i > 0:
            model.load_state_dict(torch.load(f"{self.path}/model_p{player}_i{max_i}.pt", map_location=device))
        return model

    def train_p(self, player:int):
        """
        Move models to GPU and put in train mode
        """
        self.p_models[player] = self.create_model().to(device)
        self.p_models[player].train()

    def train_s(self):
        """
        Move models to GPU and put in train mode
        """
        self.s_model = self.create_model().to(device)
        self.s_model.train()

    def save_p(self, player:int, iteration:int):
        """
        Save the model to a file
        """
        torch.save(self.p_models[player].state_dict(), f"{self.path}/model_p{player}_i{iteration}.pt")
        self.max_i_player[player] = iteration

        if os.path.exists(f"{self.path}/model_p{player}_i{iteration-1}.pt"):
            os.remove(f"{self.path}/model_p{player}_i{iteration-1}.pt")

    def save_s(self):
        """
        Save the shared model to a file
        """
        torch.save(self.s_model.state_dict(), f"{self.path}/model_s.pt")

    def load(self, player:int, iteration:int):
        """
        Load the model from a file
        """
        if os.path.exists(f"{self.path}/model_p{player}_i{iteration}.pt"):
            self.p_models[player].load_state_dict(torch.load(f"{self.path}/model_p{player}_i{iteration}.pt"))
        else:
            raise FileNotFoundError(f"Model file not found: {self.path}/model_p{player}_i{iteration}.pt")

    def get_min_iteration(self):
        """
        Get the minimum iteration across all players.
        """
        return min(self.max_i_player.values()) if self.max_i_player else 0