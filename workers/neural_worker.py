# workers/neural_worker.py

from queue import Empty
import torch
import os
import time
import traceback
from typing import Any, Dict, Optional

import utils
from core.interfaces import BaseWorker
from engines.neural_engine.runtime import NeuralRuntime


class NeuralWorker(BaseWorker):
    """
    NeuralWorker (Process Shell):
    Orchestrates data flow between IPC queues and the online AI runtime.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        queues: Dict[str, Any],
        stop_event,
    ):
        super().__init__(name, config, queues, stop_event)
        self.runtime: Optional[NeuralRuntime] = None
        self.ckpt_dir = "checkpoint"
        self.scene_name = config.get("SCENARIO_NAME", "default_scene")
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.scene_name}.ckpt")

    def _setup(self) -> None:
        self.runtime = NeuralRuntime(self.config, self.device)
        self.runtime.setup()
        self.runtime.load_checkpoint(self.ckpt_path)
        print(f"[{self.name}] Runtime setup complete on {self.device}.")

    def _loop(self) -> None:
        in_queue = self.queues["data_neural"]
        model_queue = self.queues["model"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Hyperparameters from config
        TIME_LOG = self.config.get("TIME", False)
        SAVE_INTERVAL = self.config.get("NEURAL_SAVE_INTERVAL", 50)
        NEURAL_SLEEP = self.config.get("NEURAL_SLEEP", 0.05)
        
        while not self.stop_event.is_set():
            try:
                try:
                    # 1. Fetch exactly ONE fresh CSI packet
                    raw_csi_cpu = in_queue.get(timeout=0.1)
                except Empty:
                    continue

                if self.runtime is None:
                    raise RuntimeError("Runtime not initialized")

                # 2. Push packet to runtime exactly ONCE (No NUM_EPOCHS loop)
                if TIME_LOG:
                    with utils.Timer(f"{self.name} Step"):
                        metrics_pkg = self.runtime.run_step(raw_csi_cpu)
                else:
                    metrics_pkg = self.runtime.run_step(raw_csi_cpu)

                # 3. Process results ONLY if a full batch was trained
                if metrics_pkg is not None:
                    current_step = metrics_pkg.get("step", 0)
                    loss = metrics_pkg.get("metrics", {}).get("loss", 0.0)

                    print(f"[{self.name}] Training Step #{current_step} Finished | Loss: {loss:.4f}")

                    # Sync Model Weights (Using the newly structured metrics_pkg)
                    if metrics_pkg.get("type") == "model_update":
                        try:
                            payload = {
                                "state_dict": metrics_pkg["model_state"],
                                "step": current_step,
                                "loss": loss
                            }
                            # Clear old pending models to ensure fresh weights
                            while not model_queue.empty():
                                try:
                                    model_queue.get_nowait()
                                except Empty:
                                    break
                            
                            model_queue.put(payload)
                            print(f"[{self.name}] Published updated model at step {current_step}")
                        except Exception as e:
                            print(f"[{self.name}] Sync failed: {e}")

                    # Auto-save logic
                    if current_step > 0 and current_step % SAVE_INTERVAL == 0:
                        self.runtime.save_checkpoint(self.ckpt_path)
                        print(f"[{self.name}] Checkpoint saved to {self.ckpt_path}")
                
                # Sleep to yield CPU back to OS/other workers
                time.sleep(NEURAL_SLEEP)
                in_queue.task_done()

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                traceback.print_exc()

        # Final save upon shutdown
        if self.runtime:
            self.runtime.save_checkpoint(self.ckpt_path)
            print(f"[{self.name}] Final checkpoint saved. Shutting down.")