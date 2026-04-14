# main.py

import os
import torch
import argparse
import scipy.io
import numpy as np
from multiprocessing import JoinableQueue, Queue, Event, set_start_method

import utils

from workers.symbolic_worker import SymbolicWorker
from workers.neural_worker import NeuralWorker

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

    # Symbolic is always required; neural is optional
    use_neural = config.get('USE_NEURAL', True)
    
    # Generate Reference Grid
    x_bounds = config.get("X_BOUNDS")
    y_bounds = config.get("Y_BOUNDS")

    reference_grid, x_bounds, y_bounds, x_width, y_width = \
    utils.generate_reference_grid(
        config,
        x_bounds=x_bounds,
        y_bounds=y_bounds
    )

    #DEBUG
    grid_np = reference_grid.detach().cpu().numpy()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "grid.npy"), grid_np)

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

    # 5. Setup IPC Queues
    queues = {
        'data_symbolic': JoinableQueue(), 
        'data_neural': JoinableQueue(), 
        'model': Queue(),  
        'out': Queue(),       
        'debug': Queue()
    }
    stop_event = Event()

    # 6. Initialize Workers
    print(f"[System] Initializing workers in {config.get('SYSTEM_MODE')} mode...")

    symbolic_worker = SymbolicWorker(
        name="SymbolicWorker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        reference_grid=reference_grid,
        directions_vectors=directions_vectors
    )

    if use_neural:
        neural_worker = NeuralWorker(
            name="NeuralWorker",
            config=config,
            queues=queues,
            stop_event=stop_event
        )

    # 7. Start Processes
    symbolic_worker.start()
    if use_neural:
        neural_worker.start()

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
                hmatrix_folder = os.path.join(dataset_folder, hmatrix_file)
                csi_blocks = utils.get_csi_blocks_with_cache(
                    hmatrix_folder=hmatrix_folder,
                    config=config
                )

                if not csi_blocks:
                    print(f"[System] No data in {hmatrix_file}.")
                    continue

                total_batches_in_round += len(csi_blocks)
                print(f"[System] Injecting {len(csi_blocks)} blocks from {hmatrix_file}...")
                
                for block in csi_blocks:
                    # Sends raw CSI to symbolic worker
                    queues['data_symbolic'].put(block.cpu())

            if total_batches_in_round == 0:
                print("[System] No data found in this round. Exiting.")
                break

            # --- Phase 2: Collection ---
            print(f"[System] Waiting for {total_batches_in_round} results...")
            
            # Containers for THIS round
            round_trajectory = []
            round_emission_log_probs = []
            round_posterior = []
            round_reliability = []
            round_aggregated_csi = [] # DEBUG

            received_count = 0
            while received_count < total_batches_in_round:
                try:
                    payload = queues["out"].get(timeout=None)
                    received_count += 1

                    # Data Collection
                    trajectory = utils.to_numpy(payload.get("trajectory"))
                    if trajectory is not None: round_trajectory.append(trajectory)
                    emission_log_probs_qgt = utils.to_numpy(payload.get("emission_log_probs_qgt"))
                    if emission_log_probs_qgt is not None: round_emission_log_probs.append(emission_log_probs_qgt)
                    posterior_gt = utils.to_numpy(payload.get("posterior_gt"))
                    if posterior_gt is not None: round_posterior.append(posterior_gt)
                    reliability = utils.to_numpy(payload.get("reliability"))
                    if reliability is not None: round_reliability.append(reliability)
                    # DEBUG BEGIN
                    aggregated_csi = utils.to_numpy(payload.get("aggregated_csi"))
                    if aggregated_csi is not None: round_aggregated_csi.append(aggregated_csi)
                    # DEBUG END

                except Exception as e:
                    print(f"\n[System] Collection Error: {e}")
                    break

            # --- Phase 3: Saving (Overwriting previous files) ---
            print("[System] Saving/Overwriting outputs...")

            # Saving
            if round_trajectory:
                np.save("output/predicted_trajectory.npy", np.concatenate(round_trajectory, axis=0))
            if round_emission_log_probs:
                np.save("output/emission_log_probs_qgt.npy", np.concatenate(round_emission_log_probs, axis=2))
            if round_posterior:
                np.save("output/posterior_gt.npy", np.concatenate(round_posterior, axis=1))
            if round_reliability:
                np.save("output/reliability.npy", np.stack(round_reliability, axis=0))
            # DEBUG BEGIN
            if round_aggregated_csi:
                np.save("output/aggregated_csi.npy", np.stack(round_aggregated_csi, axis=0))
            # DEBUG END

            # Queue Join
            queues['data_symbolic'].join()
            if use_neural: 
                queues['data_neural'].join()

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
        
        
        symbolic_worker.join(timeout=2)
        if symbolic_worker.is_alive(): symbolic_worker.terminate()
        if use_neural:
            neural_worker.join(timeout=2)
            if neural_worker.is_alive(): neural_worker.terminate()
        
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()