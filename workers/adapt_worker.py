# workers/adapt_worker.py

from queue import Empty
import torch
import os
import time
import traceback
from typing import Any, Dict, Optional

import utils
from core.interfaces import BaseWorker
from engines.adapt_engine.runtime import AdaptRuntime


class AdaptWorker(BaseWorker):
    """
    AdaptWorker (Process Shell):
    Orchestrates data flow between IPC queues and the AI runtime.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        queues: Dict[str, Any],
        stop_event,
    ):
        super().__init__(name, config, queues, stop_event)
        self.runtime: Optional[AdaptRuntime] = None
        self.ckpt_dir = "checkpoint"
        self.scene_name = config.get("SCENARIO_NAME", "default_scene")
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.scene_name}.ckpt")

    def _setup(self) -> None:
        self.runtime = AdaptRuntime(self.config, self.device)
        self.runtime.setup()
        self.runtime.load_checkpoint(self.ckpt_path)
        print(f"[{self.name}] Runtime setup complete on {self.device}.")

    def _loop(self) -> None:
        in_queue = self.queues["data_adapt"]
        model_queue = self.queues["model"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Hyperparameters from config
        TIME_LOG = self.config.get("TIME", False)
        SAVE_INTERVAL = self.config.get("ADAPT_SAVE_INTERVAL", 50)
        UPDATE_INTERVAL = self.config.get("ADAPT_UPDATE_INTERVAL", 1)
        ADAPT_SLEEP = self.config.get("ADAPT_SLEEP", 0.05)
        # Get Num Epochs for local training (Defaults to 1)
        NUM_EPOCHS = self.config.get("NUM_EPOCHS", 1)
        
        session_idx = 0
        while not self.stop_event.is_set():
            try:
                try:
                    raw_csi_cpu = in_queue.get(timeout=0.1)
                except Empty:
                    continue

                if self.runtime is None:
                    raise RuntimeError("Runtime not initialized")

                # --- Inner Epoch Loop ---
                # Re-train on the same data block NUM_EPOCHS times
                session_idx += 1
                for epoch_idx in range(NUM_EPOCHS):
                    if TIME_LOG:
                        with utils.Timer(f"{self.name} Step (Epoch {epoch_idx+1})"):
                            metrics_pkg = self.runtime.run_step(raw_csi_cpu)
                    else:
                        metrics_pkg = self.runtime.run_step(raw_csi_cpu)

                    # Only process if a training step was actually completed (Batch full)
                    if metrics_pkg is not None:
                        current_step = metrics_pkg.get("step", 0)
                        loss = metrics_pkg.get("metrics", {}).get("loss", 0.0)

                        # Sync Model Weights
                        if current_step % UPDATE_INTERVAL == 0:
                            try:
                                state_dict_cpu = {k: v.cpu() for k, v in self.runtime.model.state_dict().items()}
                                payload = {
                                    "state_dict": state_dict_cpu,
                                    "step": current_step,
                                    "loss": loss
                                }
                                while not model_queue.empty():
                                    try:
                                        model_queue.get_nowait()
                                    except Empty:
                                        break
                                model_queue.put(payload)
                            except Exception as e:
                                print(f"[{self.name}] Sync failed: {e}")

                        # Auto-save logic
                        if current_step % (self.runtime.update_interval * SAVE_INTERVAL) == 0:
                            self.runtime.save_checkpoint(self.ckpt_path)
                
                print(f"[{self.name}] Training #{session_idx} Finished | Loss: {loss:.4f}")

                # Sleep only after finishing all epochs for one data block
                time.sleep(ADAPT_SLEEP)
                in_queue.task_done()

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                traceback.print_exc()

        if self.runtime:
            self.runtime.save_checkpoint(self.ckpt_path)