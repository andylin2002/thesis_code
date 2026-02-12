# core/interfaces.py

import torch
from abc import ABC, abstractmethod
from multiprocessing import Process, Event, Queue

class IProcessor(ABC):
    """
    Convert raw CSI into model features.
    """
    @abstractmethod
    def extract(self, raw_csi_block):
        """
        Args:
            raw_csi_block: Raw CSI tensor for one batch.
        Returns:
            features: torch.Tensor
        """
        raise NotImplementedError

class IEstimator(ABC):
    """
    Estimate trajectory from features.
    """
    @abstractmethod
    def estimate(self, signal_data):
        """
        Args:
            features: torch.Tensor
        Returns:
            trajectory: torch.Tensor (T, 2)
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def _loop(self):
        raise NotImplementedError