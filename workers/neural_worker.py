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
        self.ckpt_dir = config.get("CHECKPOINT_DIR", "checkpoint")
        self.scene_name = config.get("SCENARIO_NAME", "default_scene")
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.scene_name}.ckpt")

    def _setup(self) -> None:
        utils.apply_reproducibility_config(self.config)
        self.runtime = NeuralRuntime(self.config, self.device)
        self.runtime.setup()
        self.runtime.load_checkpoint(self.ckpt_path)
        print(f"[{self.name}] Runtime setup complete on {self.device}.")

    def _loop(self) -> None:
        in_queue = self.queues["data_neural"]
        model_queue = self.queues["model"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Hyperparameters from config
        checkpoint_save_interval = self.config.get("CHECKPOINT_SAVE_INTERVAL", 50)
        neural_sleep = self.config.get("NEURAL_SLEEP", 0.05)
        
        while not self.stop_event.is_set():
            neural_pkg = None
            try:
                try:
                    # 1. Fetch exactly ONE fresh CSI packet
                    neural_pkg = in_queue.get(timeout=0.1)
                except Empty:
                    continue

                if self.runtime is None:
                    raise RuntimeError("Runtime not initialized")
                
                if isinstance(neural_pkg, dict) and neural_pkg.get("type") == "end_of_sequence":
                    source = neural_pkg.get("source", "unknown")
                    print(f"[{self.name}] End of CSI sequence received: {source}")

                    if self.runtime.load is not None:
                        print(f"[{self.name}] Load state before reset: {self.runtime.load.state_dict()}")

                    # Publish latest trained model to SymbolicWorker
                    if self.runtime.train is not None:
                        try:
                            state_dict = self.runtime.train.get_state_dict()
                            state_dict = self._to_cpu(state_dict)

                            while not model_queue.empty():
                                try:
                                    model_queue.get_nowait()
                                except Empty:
                                    break

                            model_queue.put({
                                "type": "model_update",
                                "state_dict": state_dict,
                                "step": self.runtime.steps,
                                "num_updates": self.runtime.num_updates,
                                "source": source,
                            })

                            print(f"[{self.name}] Published latest model at end of sequence: {source}")

                        except Exception as e:
                            print(f"[{self.name}] End-of-sequence publish failed: {e}")

                    self.runtime.save_checkpoint(self.ckpt_path)
                    self.runtime.reset_sequence()
                    continue

                # 2. Push packet to runtime exactly ONCE (No NUM_EPOCHS loop)
                metrics_pkg = self.runtime.run_step(neural_pkg)

                # 3. Process results ONLY if a full batch was trained
                if metrics_pkg is not None:
                    current_step = metrics_pkg.get("step", 0)
                    loss = metrics_pkg.get("metrics", {}).get("loss", 0.0)

                    print(f"[{self.name}] Training Step #{current_step} Finished | Loss: {loss:.4f}")

                    # Sync Model Weights (Using the newly structured metrics_pkg)
                    if metrics_pkg.get("type") == "model_update":
                        try:
                            payload = {
                                "state_dict": self._to_cpu(metrics_pkg["state_dict"]),
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
                    if current_step > 0 and current_step % checkpoint_save_interval == 0:
                        self.runtime.save_checkpoint(self.ckpt_path)
                        print(f"[{self.name}] Checkpoint saved to {self.ckpt_path}")
                
                # Sleep to yield CPU back to OS/other workers
                time.sleep(neural_sleep)

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                traceback.print_exc()

            finally:
                if neural_pkg is not None:
                    in_queue.task_done()

        # Final save upon shutdown
        if self.runtime:
            self.runtime.save_checkpoint(self.ckpt_path)
            print(f"[{self.name}] Final checkpoint saved. Shutting down.")

    def _to_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cpu(v) for v in obj)
        return obj