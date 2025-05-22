
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from memory.memory import Memory

device = torch.device("mps" if torch.mps.is_available() else "cpu")

class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer, loss_fn, memory:Memory, n_samples: int, iteration: int):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = InfoDataset(memory, n_samples)
        self.iteration = iteration
        

    def train(self, epochs, batch_size):
        for epoch in range(epochs):

            data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            self.model.train()
            running_loss = 0.0
            with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch, advantages, t in data_loader:
                    loss_weight = t / self.iteration
                    
                    self.optimizer.zero_grad()

                    cards, bet_feats = batch
                    pred = self.model(cards, bet_feats)

                    loss = self.loss_fn(pred, advantages.float().to(device), loss_weight.unsqueeze(1).to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    pbar.update(1)
                    running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))



class InfoDataset(Dataset):
    def __init__(self, memory:Memory, n):
        self.data = memory.sample(n)

    def __len__(self):
        return len(list(self.data))

    def __getitem__(self, idx):
        return self.data[idx]