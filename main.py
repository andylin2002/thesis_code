# main.py

import os
import torch
import argparse
import scipy.io
import numpy as np
from multiprocessing import JoinableQueue, Queue, Event, set_start_method

import utils

from workers.infer_worker import InferWorker
from workers.adapt_worker import AdaptWorker

# Configuration
CONFIG_PATH = 'config.yaml'
CHECKPOINT_DIR = 'checkpoint'
OUTPUT_DIR = 'output'
DATASET_ROOT = 'dataset'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Setup Multiprocessing
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
    x_bounds = config.get("X_BOUNDS")
    y_bounds = config.get("Y_BOUNDS")

    reference_grid, x_bounds, y_bounds, x_width, y_width = \
    utils.generate_reference_grid(
        config,
        x_bounds=x_bounds,
        y_bounds=y_bounds
    )

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

    # Prepare Checkpoint & Output
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    hmatrix_list = config.get('HMATRIX_LIST', [])

    # Save Grid for Debug
    grid_np = utils.to_numpy(reference_grid)
    if grid_np is not None:
        np.save("output/grid.npy", grid_np)
        print(f"[IO] Reference Grid saved. Shape: {grid_np.shape}")

    # 5. Setup IPC Queues
    queues = {
        'data_infer': JoinableQueue(), 
        'data_adapt': JoinableQueue(), 
        'model': Queue(),  
        'out': Queue(),       
        'debug': Queue()
    }
    stop_event = Event()

    # 6. Initialize Workers
    print(f"[System] Initializing workers in {config.get('SYSTEM_MODE')} mode...")
    
    infer_worker = InferWorker(
        name="InferWorker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        reference_grid=reference_grid,
        directions_vectors=directions_vectors
    )

    adapt_worker = AdaptWorker(
        name="AdaptWorker",
        config=config,
        queues=queues,
        stop_event=stop_event
    )

    # 7. Start Processes
    infer_worker.start()
    adapt_worker.start()

    # 8. Main Loop (Modified for iterative Injection -> Collection -> Saving)
    loop_mode = config.get('LOOP', False)
    round_idx = 0

    try:
        while True:
            round_idx += 1
            print(f"\n{'='*50}")
            print(f"[System] Starting Round {round_idx}")
            print(f"{'='*50}")
            
            # --- Phase 1: Injection ---
            total_batches_in_round = 0
            
            for hmatrix_file in hmatrix_list:
                hmatrix_path = os.path.join(dataset_folder, hmatrix_file)
                csi_blocks = utils.load_and_preprocess_csi_dataset(
                    Hmatrix=hmatrix_path,
                    config=config
                )

                if not csi_blocks:
                    print(f"[System] No data in {hmatrix_file}.")
                    continue

                total_batches_in_round += len(csi_blocks)
                print(f"[System] Injecting {len(csi_blocks)} blocks from {hmatrix_file}...")
                
                for block in csi_blocks:
                    # Send CPU tensor to avoid CUDA IPC locks
                    queues['data_infer'].put(block.cpu())
                    queues['data_adapt'].put(block.cpu())

            if total_batches_in_round == 0:
                print("[System] No data found in this round. Exiting.")
                break

            # --- Phase 2: Collection ---
            print(f"[System] Waiting for {total_batches_in_round} results...")
            
            # Containers for THIS round
            round_trajectories = [] 
            round_epds = []
            round_stpds = []
            round_tpds = []
            round_egs = []
            round_tgs = []

            received_count = 0
            while received_count < total_batches_in_round:
                try:
                    payload = queues["out"].get(timeout=None)
                    received_count += 1

                    # 1. Collect Trajectory
                    traj = utils.to_numpy(payload.get("trajectory"))
                    if traj is not None: round_trajectories.append(traj)

                    # 2. Collect EPD
                    epd = utils.to_numpy(payload.get("epd"))
                    if epd is not None: round_epds.append(epd)

                    # 3. Collect STPD
                    stpd = utils.to_numpy(payload.get("stpd"))
                    if stpd is not None: round_stpds.append(stpd)

                    # 4. Collect TPD
                    tpd = utils.to_numpy(payload.get("tpd"))
                    if tpd is not None: round_tpds.append(tpd)

                    # 5. Collect Gating
                    eg = utils.to_numpy(payload.get("emission_gating"))
                    if eg is not None: round_egs.append(eg)

                    tg = utils.to_numpy(payload.get("transition_gating"))
                    if tg is not None: round_tgs.append(tg)

                except Exception as e:
                    print(f"\n[System] Collection Error: {e}")
                    break

            # Wait for workers to finish processing tasks
            queues['data_infer'].join()
            queues['data_adapt'].join()
            print(f"\n[System] Round {round_idx} processing complete.")

            # --- Phase 3: Saving (Overwriting previous files) ---
            print("[System] Saving/Overwriting outputs...")

            # Save Trajectory
            if round_trajectories:
                full_path = np.concatenate(round_trajectories, axis=0)
                np.save("output/predicted_trajectory.npy", full_path)
            
            # Save EPD
            if round_epds:
                try:
                    full_epd = np.concatenate(round_epds, axis=1)
                    np.save("output/epd.npy", full_epd)
                except ValueError: pass

            # Save STPD
            if round_stpds:
                try:
                    full_stpd = np.concatenate(round_stpds, axis=1)
                    np.save("output/stpd.npy", full_stpd)
                except ValueError: pass

            # Save TPD
            if round_tpds:
                try:
                    full_tpd = np.concatenate(round_tpds, axis=0)
                    np.save("output/tpd.npy", full_tpd)
                except ValueError: pass

            # Save Emission Gating
            if round_egs:
                try:
                    full_eg = np.stack(round_egs, axis=0)
                    np.save("output/emission_gating.npy", full_eg)
                except ValueError: pass

            # Save Transition Gating
            if round_tgs:
                try:
                    full_tg = np.stack(round_tgs, axis=0)
                    np.save("output/transition_gating.npy", full_tg)
                except ValueError: pass

            print(f"[IO] Outputs for Round {round_idx} saved to disk.")

            if not loop_mode:
                break

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    
    finally:
        # 9. Graceful Shutdown
        print("[System] Shutting down workers...")
        stop_event.set()

        while not queues['out'].empty(): queues['out'].get()
        while not queues['model'].empty(): queues['model'].get()
        
        infer_worker.join(timeout=2)
        adapt_worker.join(timeout=2)

        if infer_worker.is_alive(): infer_worker.terminate()
        if adapt_worker.is_alive(): adapt_worker.terminate()
        
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()