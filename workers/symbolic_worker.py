# workers/symbolic_worker.py

from queue import Empty

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
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        self.runtime = None

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
        in_queue = self.queues["data_symbolic"]               # Raw CSI input (CPU tensors)
        out_queue = self.queues["out"]                     # Output to main
        debug_queue = self.queues.get("debug", None)       # Optional debug

        TIME = self.config.get("TIME", False)

        session_idx = 0
        while not self.stop_event.is_set():
            try:
                raw_csi_block = in_queue.get(timeout=0.1)
            except Empty:
                continue
            
            session_idx += 1
            try:
                # One step inference (returns CPU-safe dict)
                if TIME:
                    with utils.Timer(f"{self.name} Inference"):
                        pkg_cpu = self.runtime.step(raw_csi_block)
                else:
                    pkg_cpu = self.runtime.step(raw_csi_block)

                # Always send main output
                out_queue.put(pkg_cpu)
                print(f"[{self.name}] Save #{session_idx} prediction trajectory!")

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
