import joblib
import numpy as np
from torch import Tensor


class Memory:
    def __init__(self, size: int, path: str = None):
        self.size = size
        self.infoset_memory = []
        self.action_probs_memory = []
        self.timestep_memory = []

        if path is not None:
            self.load(path)

    def __len__(self):
        return len(self.infoset_memory)


    def store(self, infoset, action_probs:Tensor, t:int):
        if len(self.infoset_memory) >= self.size:
            raise MemoryError("Memory is full. TODO: implement reservoir sampling.")
            return
        self.infoset_memory.append(infoset)
        self.action_probs_memory.append(action_probs)
        self.timestep_memory.append(t)

    def sample(self, n):
        size = min(n, len(self.infoset_memory))

        indices = np.random.choice(len(self.infoset_memory), size=size, replace=True)
        return [(self.infoset_memory[i], self.action_probs_memory[i], self.timestep_memory[i]) for i in indices]

    def max_timestep(self):
        return self.timestep_memory[-1] if self.timestep_memory else 0
    
    def save(self, path: str):
        joblib.dump({
            "i": self.infoset_memory,
            "a": self.action_probs_memory,
            "t": self.timestep_memory
        }, path, compress=3)

    def load(self, path: str):
        data = joblib.load(path)
        self.infoset_memory = data["i"]
        self.action_probs_memory = data["a"]
        self.timestep_memory = data["t"]




        

