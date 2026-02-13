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

        # Checkpoint paths
        self.ckpt_dir = "checkpoint"
        self.scene_name = config.get("SCENARIO_NAME", "default_scene")
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.scene_name}.ckpt")

    def _setup(self) -> None:
        """Initialize the 3-stage runtime (Load -> Represent -> Train)."""
        # Create the runtime instance. This initializes the model and optimizer.
        self.runtime = AdaptRuntime(self.config, self.device)
        self.runtime.setup()
        self.runtime.load_checkpoint(self.ckpt_path)
        print(f"[{self.name}] Runtime setup complete on {self.device}.")

    def _loop(self) -> None:
            in_queue = self.queues["data_adapt"]
            model_queue = self.queues["model"]

            os.makedirs(self.ckpt_dir, exist_ok=True)
            
            # Hyperparameters
            TIME = self.config.get("TIME", False)
            SAVE_INTERVAL = self.config.get("ADAPT_SAVE_INTERVAL", 50)
            UPDATE_INTERVAL = self.config.get("ADAPT_UPDATE_INTERVAL", 1) # Sync interval
            ADAPT_SLEEP = self.config.get("ADAPT_SLEEP", 0.05)

            while not self.stop_event.is_set():
                try:
                    try:
                        raw_csi_cpu = in_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    if self.runtime is None:
                        raise RuntimeError("Runtime not initialized")

                    # Step runtime (Training)
                    if TIME:
                        with utils.Timer(f"{self.name} Total Step"):
                            metrics_pkg = self.runtime.run_step(raw_csi_cpu)
                    else:
                        metrics_pkg = self.runtime.run_step(raw_csi_cpu)

                    # Process Output
                    if metrics_pkg is not None:
                        current_step = metrics_pkg.get("step", 0)
                        loss = metrics_pkg.get("metrics", {}).get("loss", 0.0)
                        print(f"[{self.name}] Step {current_step} | Loss: {loss:.4f}", flush=True)

                        # [CRITICAL] Sync Model Weights to InferWorker
                        if current_step % UPDATE_INTERVAL == 0:
                            try:
                                # 1. Extract weights
                                state_dict = self.runtime.model.state_dict()
                                
                                # 2. Convert to CPU (Required for IPC)
                                state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
                                
                                # 3. Payload
                                payload = {
                                    "state_dict": state_dict_cpu,
                                    "step": current_step,
                                    "loss": loss
                                }
                                
                                # 4. Send (Non-blocking, keep only latest)
                                while not model_queue.empty():
                                    try:
                                        model_queue.get_nowait()
                                    except Empty:
                                        break
                                
                                model_queue.put(payload)

                            except Exception as e:
                                print(f"[{self.name}] Failed to send model update: {e}")

                        # Auto-save
                        if current_step % (self.runtime.update_interval * SAVE_INTERVAL) == 0:
                            print(f"[{self.name}] Auto-saving checkpoint...")
                            self.runtime.save_checkpoint(self.ckpt_path)
                        
                        time.sleep(ADAPT_SLEEP)

                    in_queue.task_done()

                except Exception as e:
                    print(f"[{self.name}] Error: {e}")
                    traceback.print_exc()

            # Final save
            if self.runtime:
                print(f"[{self.name}] Worker stopping. Saving final checkpoint...")
                self.runtime.save_checkpoint(self.ckpt_path)