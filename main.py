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

    # Prepare Checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    scene_name = config.get('SCENARIO_NAME', 'Untitled_Scene')
    hmatrix_list = config.get('HMATRIX_LIST', [])
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{scene_name}.ckpt")

    # DEBUG OUTPUT
    grid_np = utils.to_numpy(reference_grid)
    if grid_np is not None:
        np.save("output/grid.npy", grid_np)
        print(f"[IO] Reference Grid saved. Shape: {grid_np.shape}")

    # 5. Setup IPC Queues
    # 'data': JoinableQueue allows .join() to wait for task completion
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

    # 8. Main Loop
    all_trajectories = [] 
    all_epds = []
    all_stpds = []
    all_tpds = []

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
                    queues['data_infer'].put(block.cpu())
                    queues['data_adapt'].put(block.cpu())

            if not loop_mode:
                break
        
        # =========================================================
        # Synchronization: Wait for all tasks to complete
        # =========================================================
        print(f"[System] Injection done. Waiting for {total_batches_sent} batches...")
        
        # Loop until we receive exactly the number of batches sent
        received_count = 0
        while received_count < total_batches_sent:
            try:
                payload = queues["out"].get(timeout=None)  # dict from InferWorker
                received_count += 1

                # 1. Collect Trajectory
                traj = utils.to_numpy(payload.get("trajectory"))
                if traj is not None:
                    all_trajectories.append(traj)

                # 2. Collect EPD
                epd = utils.to_numpy(payload.get("epd"))
                if epd is not None:
                    all_epds.append(epd)

                # 3. Collect STPD
                stpd = utils.to_numpy(payload.get("stpd"))
                if stpd is not None:
                    all_stpds.append(stpd)

                # 4. Collect TPD
                tpd = utils.to_numpy(payload.get("tpd"))
                if tpd is not None:
                    all_tpds.append(tpd)

                print(f"[System] Progress: {len(all_trajectories)}/{total_batches_sent} saved.", flush=True)

            except Queue.Empty:
                continue

            except Exception as e:
                print(f"\n[System] Collection Error: {e}")
                break

        queues['data_infer'].join()
        print("[System] InferWorker tasks cleared.")

        queues['data_adapt'].join()
        print("[System] AdaptWorker tasks cleared.")

        # Save Trajectory
        if all_trajectories:
            full_path = np.concatenate(all_trajectories, axis=0)
            np.save("output/predicted_trajectory.npy", full_path)
            print(f"[IO] Trajectory saved. Shape: {full_path.shape}")

        # Save EPD
        if all_epds:
            try:
                full_epd = np.concatenate(all_epds, axis=1)
                np.save("output/epd.npy", full_epd)
                print(f"[IO] EPD saved. Shape: {full_epd.shape}")
            except ValueError as e:
                print(f"[IO] Error merging EPD: {e}")

        # Save STPD
        if all_stpds:
            try:
                full_stpd = np.concatenate(all_stpds, axis=1)
                np.save("output/stpd.npy", full_stpd)
                print(f"[IO] STPD saved. Shape: {full_stpd.shape}")
            except ValueError as e:
                print(f"[IO] Error merging STPD: {e}")

        # Save TPD
        if all_tpds:
            try:
                full_tpd = np.concatenate(all_tpds, axis=0)
                np.save("output/tpd.npy", full_tpd)
                print(f"[IO] TPD saved. Shape: {full_tpd.shape}")
            except ValueError as e:
                print(f"[IO] Error merging TPD: {e}")

        print("\n[System] All tasks processed.")

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