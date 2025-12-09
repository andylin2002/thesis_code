import os
import time
import torch
import argparse
import scipy.io
import numpy as np
from multiprocessing import JoinableQueue, Queue, Event, set_start_method
from queue import Empty

# Import Utility
import utils

# Import New Workers
from workers.csi_worker import CSI_Worker
from workers.tfm_worker import TFM_Worker

# Configuration Paths
CONFIG_PATH = 'config.yaml'
CHECKPOINT_DIR = 'checkpoint'
DATASET_ROOT = 'dataset'
# Set Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Setup Multiprocessing (Spawn method required for CUDA)
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
        print("[System] Configuration loading failed.")
        return
    
    # Load Environment Config
    dataset_folder = os.path.join(DATASET_ROOT, config['DATASET_FOLDER'])
    env_config_path = os.path.join(dataset_folder, config['ENV_CONFIG'])
    env_config = utils.load_yaml_config(env_config_path)
    if not env_config:
        print("[System] Environment config loading failed.")
        return
    config.update(env_config)

    # Update Config with Args
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

    # Prepare Checkpoint Path
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    scene_name = config.get('SCENARIO_NAME', 'Untitled_Scene')
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{scene_name}.ckpt")

    # 5. Setup IPC Queues
    queues = {
        'data': JoinableQueue(), # Input: Raw CSI
        'result': Queue(),       # Output: Trajectory
        'model': Queue()         # Update: Model Weights
    }
    stop_event = Event()

    # 6. Initialize Workers
    print(f"[System] Initializing workers in {config.get('SYSTEM_MODE')} mode...")
    
    # CSI Worker (Inference)
    csi_worker = CSI_Worker(
        name="CSI_Worker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        reference_grid=reference_grid,
        directions_vectors=directions_vectors
    )

    # AI Worker (Training)
    tfm_worker = TFM_Worker(
        name="AI_Worker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        directions_vectors=directions_vectors,
        checkpoint_path=ckpt_path
    )

    # 7. Start Processes
    csi_worker.start()
    tfm_worker.start()

    # 8. Data Injection & Collection Loop
    hmatrix_list = config.get('HMATRIX_LIST', [])
    all_trajectories = [] # Buffer to store results

    try:
        loop_mode = config.get('LOOP', False)
        while True:
            for hmatrix_file in hmatrix_list:
                hmatrix_path = os.path.join(dataset_folder, hmatrix_file)
                
                # Load CSI Data
                csi_blocks = utils.load_and_preprocess_csi_dataset(
                    Hmatrix=hmatrix_path,
                    config=config,
                    device=DEVICE
                )

                if not csi_blocks:
                    print("[System] No data loaded.")
                    break

                print(f"\n[System] Injecting {len(csi_blocks)} CSI blocks...")
                
                for block in csi_blocks:
                    # A. Inject Data
                    queues['data'].put(block)
                    
                    # B. Collect Results (Non-blocking check)
                    while not queues['result'].empty():
                        try:
                            res = queues['result'].get_nowait()
                            
                            # Parse result (Handle Dict from Proposed or Tensor from Baseline)
                            traj = res['pseudo_gt'] if isinstance(res, dict) and 'pseudo_gt' in res else res
                            
                            if isinstance(traj, torch.Tensor):
                                traj = traj.detach().cpu().numpy()
                            
                            all_trajectories.append(traj)
                        except Empty:
                            break
                        except Exception as e:
                            print(f"[System] Collection error: {e}")

            if not loop_mode:
                break
        
        # Flush remaining results after injection is done
        print("[System] Waiting for remaining results...")
        time_waited = 0
        while time_waited < 5.0: # 5 seconds timeout
            if not queues['result'].empty():
                res = queues['result'].get()
                traj = res['pseudo_gt'] if isinstance(res, dict) and 'pseudo_gt' in res else res
                if isinstance(traj, torch.Tensor):
                    traj = traj.detach().cpu().numpy()
                all_trajectories.append(traj)
                time_waited = 0 # Reset timeout if data received
            else:
                time.sleep(0.1)
                time_waited += 0.1

        # Save Final Trajectory
        if all_trajectories:
            full_path = np.concatenate(all_trajectories, axis=0)
            os.makedirs("output", exist_ok=True)
            save_path = os.path.join("output", "predicted_trajectory.npy")
            np.save(save_path, full_path)
            print(f"[System] Trajectory saved to: {save_path} (Shape: {full_path.shape})")
        else:
            print("[System] No trajectory data collected.")

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    
    finally:
        # 9. Graceful Shutdown
        stop_event.set()
        
        tfm_worker.join(timeout=2)
        csi_worker.join(timeout=2)

        if tfm_worker.is_alive(): tfm_worker.terminate()
        if csi_worker.is_alive(): csi_worker.terminate()
        
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()