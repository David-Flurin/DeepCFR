import logging
import os

from .memory import Memory

log = logging.getLogger(__name__)

class MemoryManager:
    """
    Manage infoset memory for multiple players and strategy memory.
    """
    def __init__(self, n_players: int, memory_size: int, save_path: str):
        self.save_path = save_path
        self.max_i_player = {}

        self.p_memory: dict[str, Memory] = {}
        for p in range(n_players):
            filename = f"{save_path}/memory_p{p}.joblib"
            log.debug(f"Loading memory for player {p} from {filename}")
            if os.path.exists(filename):
                self.p_memory[p] = Memory(memory_size, filename)
                self.max_i_player[p] = self.p_memory[p].max_timestep()
                
            else:
                self.p_memory[p] = Memory(memory_size)
                self.max_i_player[p] = 0
        print("Initialized player memory for players", self.max_i_player)

        filename = f"{save_path}/memory_s.joblib"
        self.s_memory = Memory(memory_size, filename if os.path.exists(filename) else None)

        log.info(f"Loaded memory for {n_players} players from {save_path}. Min iteration: {self.get_min_iteration()}")

    def store_p(self, p: int, infoset, adv, t):
        """
        Store an infoset for a player.
        """
        self.p_memory[p].store(infoset, adv, t)
        if self.max_i_player[p] < t:
            self.max_i_player[p] = t

    def store_s(self, infoset, action_probs, t):
        """
        Store an infoset for the strategy.
        """
        self.s_memory.store(infoset, action_probs, t)

    def dump_p(self, p: int):
        """
        Dump the memory for a player to a file.
        """
        filename = f"{self.save_path}/memory_p{p}.joblib"
        self.p_memory[p].save(filename)

    def dump_s(self):
        """
        Dump the strategy memory to a file.
        """
        filename = f"{self.save_path}/memory_s.joblib"
        self.s_memory.save(filename)

    def length_p(self, p: int):
        """
        Get the length of the memory for a player.
        """
        return len(self.p_memory[p])
    
    def length_s(self):
        """
        Get the length of the strategy memory.
        """
        return len(self.s_memory)

    def get_min_iteration(self):
        """
        Get the minimum iteration across all players.
        """
        return min(self.max_i_player.values())
    
    def get_min_player(self):
        """
        Get the player with the minimum iteration.
        """
        return min(self.max_i_player, key=self.max_i_player.get)
