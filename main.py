#main.py

import os
import time
import torch
import argparse
import scipy.io
import numpy as np
from multiprocessing import JoinableQueue, Queue, Event, set_start_method
from queue import Empty

import utils

from workers.infer_worker import INFER_Worker
# from workers.adapt_worker import ADAPT_Worker

# Configuration
CONFIG_PATH = 'config.yaml'
CHECKPOINT_DIR = 'checkpoint'
DATASET_ROOT = 'dataset'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Setup Multiprocessing
    # 'spawn' is required for CUDA compatibility
    try:
        if torch.cuda.is_available():
            set_start_method('spawn', force=True)
            print("[System] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if 'start method has been set' not in str(e):
             raise e

    # 2. Parse Arguments
    parser = argparse.ArgumentParser(description='CSI Indoor Localization System')
    parser.add_argument('--em_max_iter', type=int, default=100)
    parser.add_argument('--round', type=int, default=5000)
    args = parser.parse_args()

    # 3. Load Configuration
    config = utils.load_yaml_config(CONFIG_PATH)
    if not config:
        print("[System] Config loading failed.")
        return
    
    # Load Environment Config
    dataset_folder = os.path.join(DATASET_ROOT, config['DATASET_FOLDER'])
    env_config = utils.load_yaml_config(os.path.join(dataset_folder, config['ENV_CONFIG']))
    if not env_config:
        print("[System] Env config loading failed.")
        return
    config.update(env_config)
    
    # Override with Args
    config['EM_MAX_ITER'] = args.em_max_iter
    config['ROUND'] = args.round
    
    # Generate Reference Grid
    reference_grid, x_bounds, y_bounds, x_width, y_width = utils.generate_reference_grid(config)
    config['X_BOUNDS'] = x_bounds
    config['Y_BOUNDS'] = y_bounds
    config['X_WIDTH'] = x_width
    config['Y_WIDTH'] = y_width

    # 4. Load Static Resources
    try:
        mat = scipy.io.loadmat("directions.mat")
        directions_vectors = mat['directions']
        print("[System] directions.mat loaded successfully.")
    except Exception as e:
        print(f"[System] Failed to load directions.mat: {e}")
        return

    # Prepare Checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    scene_name = config.get('SCENARIO_NAME', 'Untitled_Scene')
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{scene_name}.ckpt")

    # 5. Setup IPC Queues
    # 'data': JoinableQueue allows .join() to wait for task completion
    queues = {
        'data': JoinableQueue(), 
        'result': Queue(),       
        'model': Queue(), 
        'debug': Queue()
    }
    stop_event = Event()

    # 6. Initialize Workers
    print(f"[System] Initializing workers in {config.get('SYSTEM_MODE')} mode...")
    
    infer_worker = INFER_Worker(
        name="INFER_Worker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        reference_grid=reference_grid,
        directions_vectors=directions_vectors
    )

    # adapt_worker = ADAPT_Worker(
    #     name="ADAPT_Worker",
    #     config=config,
    #     queues=queues,
    #     stop_event=stop_event,
    #     directions_vectors=directions_vectors,
    #     checkpoint_path=ckpt_path
    # )

    # 7. Start Processes
    infer_worker.start()
    # adapt_worker.start()

    # 8. Main Loop
    hmatrix_list = config.get('HMATRIX_LIST', [])
    all_trajectories = [] 

    total_batches_sent = 0
    try:
        loop_mode = config.get('LOOP', False)
        while True:
            for hmatrix_file in hmatrix_list:
                hmatrix_path = os.path.join(dataset_folder, hmatrix_file)
                
                # Load CSI Data
                csi_blocks = utils.load_and_preprocess_csi_dataset(
                    Hmatrix=hmatrix_path,
                    config=config
                )

                if not csi_blocks:
                    print("[System] No data loaded.")
                    break

                # Update counter
                total_batches_sent += len(csi_blocks)
                print(f"\n[System] Injecting {len(csi_blocks)} blocks. Total sent: {total_batches_sent}")
                
                for block in csi_blocks:
                    # Inject Data
                    # Important: Send CPU tensor to avoid CUDA IPC locks
                    queues['data'].put(block.cpu())

            if not loop_mode:
                break
        
        # =========================================================
        # Synchronization: Wait for all tasks to complete
        # =========================================================
        print(f"[System] Injection done. Waiting for {total_batches_sent} batches...")
        queues["data"].join()
        print("[System] infer_worker finished processing all injected blocks.")
        
        # Loop until we receive exactly the number of batches sent
        while len(all_trajectories) < total_batches_sent:
            try:
                # Listen to 'save' queue. ADAPT worker listens to 'result'.
                res_path = queues['debug'].get(timeout=None) 
                
                # Convert to Numpy
                if hasattr(res_path, 'detach'):
                    res_path = res_path.detach().cpu().numpy()
                
                all_trajectories.append(res_path)
                
                # Instant Save (Overwrite .npy)
                full_path = np.concatenate(all_trajectories, axis=0)
                os.makedirs("output", exist_ok=True)
                np.save("output/predicted_trajectory.npy", full_path)
                
                print(f"\r[System] Progress: {len(all_trajectories)}/{total_batches_sent} saved.", end="")
            
            except Exception as e:
                print(f"\n[System] Collection Error: {e}")
                break

        print("\n[System] All tasks processed.")

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    
    finally:
        # 9. Graceful Shutdown
        print("[System] Shutting down workers...")
        stop_event.set()
        
        # adapt_worker.join(timeout=2)
        infer_worker.join(timeout=2)

        # if adapt_worker.is_alive(): adapt_worker.terminate()
        if infer_worker.is_alive(): infer_worker.terminate()
        
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()