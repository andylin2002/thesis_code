# workers/symbolic_worker.py

from queue import Empty
from typing import Optional

import utils
from core.interfaces import BaseWorker
from engines.symbolic_engine.runtime import SymbolicRuntime


class SymbolicWorker(BaseWorker):
    """
    Thin worker:
    - handles queue IO + lifecycle
    - delegates inference orchestration to SymbolicRuntime
    """

    def __init__(self, name, config, queues, stop_event, reference_grid, directions_vectors):
        super().__init__(name, config, queues, stop_event)
        self.runtime: Optional[SymbolicRuntime] = None
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

    def _setup(self):
        model_queue = self.queues.get("model", None)

        self.runtime = SymbolicRuntime(
            config=self.config,
            device=self.device,
            reference_grid=self.reference_grid,
            directions_vectors=self.directions_vectors,
            model_queue=model_queue
        )

        print(f"[{self.name}] Setup complete. Runtime loaded.")

    def _loop(self):
        in_queue = self.queues["data_symbolic"]
        out_queue = self.queues["out"]
        neural_queue = self.queues.get("data_neural", None)
        debug_queue = self.queues.get("debug", None)       # Optional debug

        enable_neural_training = self.config.get("ENABLE_NEURAL_TRAINING", True)

        session_idx = 0
        while not self.stop_event.is_set():
            try:
                raw_csi_block = in_queue.get(timeout=0.1)
                if isinstance(raw_csi_block, dict) and raw_csi_block.get("type") == "end_of_sequence":
                    if enable_neural_training and neural_queue is not None:
                        neural_queue.put(raw_csi_block)

                    in_queue.task_done()
                    continue
            except Empty:
                continue
            
            session_idx += 1
            try:
                # One step inference (returns CPU-safe dict)
                pkg_cpu = self.runtime.step(raw_csi_block)

                # Always send main output
                out_queue.put(pkg_cpu)
                print(f"[{self.name}] Save #{session_idx} outputs!")

                if enable_neural_training and neural_queue is not None:
                    neural_pkg = {
                        "aggregated_csi": pkg_cpu["aggregated_csi"],
                        "emission_log_probs_qgt": pkg_cpu["emission_log_probs_qgt"],
                        "posterior_gt": pkg_cpu["posterior_gt"],
                    }
                    neural_queue.put(neural_pkg)

                # Optional debug (non-blocking)
                if debug_queue is not None:
                    try:
                        debug_queue.put_nowait(pkg_cpu["trajectory"])
                    except Exception:
                        # debug queue full -> drop safely
                        pass

            except Exception as e:
                print(f"[{self.name}] Error in loop processing: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # IMPORTANT: always release JoinableQueue.join() in main.py
                in_queue.task_done()
