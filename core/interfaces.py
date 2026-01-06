# core/interfaces.py

import torch
from abc import ABC, abstractmethod
from multiprocessing import Process, Event, Queue

class ISignalProcessor(ABC):
    """
    Interface for signal processing strategies.
    Responsible for converting raw CSI into features.
    """
    @abstractmethod
    def extract(self, raw_csi_block):
        """
        Args:
            raw_csi_block: Raw CSI data chunk from the queue.
        
        Returns:
            dict: Processed data based on mode.
                  - Baseline: {'mode': 'BASELINE', 'features': ...}
                  - Proposed: {'mode': 'PROPOSED', 'features': ..., 'spd': ...}
        """
        pass

class ILocationEstimator(ABC):
    """
    Interface for location estimation strategies.
    Responsible for estimating trajectories using features.
    """
    @abstractmethod
    def estimate(self, signal_data):
        """
        Args:
            signal_data (dict): The output dictionary from ISignalProcessor.extract().
            
        Returns:
            Tensor/np.array: The predicted path coordinates.
        """
        pass

class BaseWorker(Process, ABC):
    """
    Abstract base class for multiprocessing workers.
    Handles process lifecycle (start/stop/init); subclasses must implement `_loop`.
    """
    def __init__(self, name, config, queues, stop_event: Event): # type: ignore
        super().__init__(name=name)
        self.config = config
        self.queues = queues
        self.stop_event = stop_event

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config['device'] = self.device

    def run(self):
        """
        Main entry point for the process. 
        Do NOT override this. Implement `_setup` and `_loop` instead.
        """
        print(f"[{self.name}] Initializing on {self.device}...")
        try:
            self._setup()
            print(f"[{self.name}] Running loop...")
            self._loop()
        except Exception as e:
            print(f"[{self.name}] CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[{self.name}] Stopped.")

    @abstractmethod
    def _setup(self):
        """
        Perform heavy initialization here (e.g., loading models, creating buffers).
        Runs inside the child process to ensure memory isolation.
        """
        pass

    @abstractmethod
    def _loop(self):
        """
        The main execution loop. Should monitor `self.stop_event`.
        """
        pass