import logging
import os

import torch

device = torch.device("mps" if torch.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, n_players, create_model: callable, save_path: str):
        self.p_models = {}
        self.s_model = create_model().to(device)
        self.create_model = create_model
        self.n_players = n_players
        self.path = save_path

        self.max_i_player = {}

    def load_latest_available_model_iteration(self, p: int):
        """
        Find the latest available model iteration for a given player.
        """
        model_files = [f for f in os.listdir(self.path) if f.startswith(f"model_p{p}_i") and f.endswith(".pt")]
        if not model_files:
            return 0
        max_i = max(int(f.split('_i')[-1].split('.pt')[0]) for f in model_files)

        self.p_models[p].load_state_dict(torch.load(f"{self.path}/model_p{p}_i{max_i}.pt"))
        return max_i
    
    def get_p_model(self, player:int):
        """
        Get the model for a given player.
        """
        return self.p_models[player]

    def eval(self):
        """
        Move models to CPU and put in eval mode
        """
        for p in range(self.n_players):
            self.p_models[p] = self.create_model().to('cpu')
            log.debug(f"Creating model for player {p}")
            iteration = self.load_latest_available_model_iteration(p)
            self.max_i_player[p] = iteration
            self.p_models[p].eval()

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