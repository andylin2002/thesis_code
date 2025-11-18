import utils
import torch
import argparse
from typing import List, Dict, Any
import time
from multiprocessing import Queue, Process, set_start_method, Event
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F, numpy as np, scipy.io, os

from csi2traj import run_csi2traj
from transformer.transformer_tool import convert_long_trajectory_to_ids, create_transformer_instance
from transformer.architecture.noam_opt import NoamOpt
from transformer.architecture.batch import subsequent_mask

DATASET_FOLDER = 'dataset_1'
CHECKPOINT_DIR = 'checkpoint'
CONFIG_PATH = 'config.yaml'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_DIRECTIONS = 9


##############################################
############# --- Controller --- #############
##############################################

def main():
##### --- Set multiprocessing start method to 'spawn' to ensure safe CUDA usage in subprocesses.
    try:
        if torch.cuda.is_available():
            set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn' for CUDA compatibility.")
    except RuntimeError as e:
        if 'start method has been set' not in str(e):
             raise e

##### --- Initialize the argument parser
    parser=argparse.ArgumentParser(description='CSI Indoor Position System Parameter')
    parser.add_argument('--em_max_iter', type=int, default=20)
    parser.add_argument('--round', type=int, default=5000)

    args=parser.parse_args()

##### --- Loading Environment & System Configuration ---
    config = utils.load_yaml_config(CONFIG_PATH)
    if not (config):
        print("Configuration loading failed.")
        return
    
##### --- Reference Point Setup ---
    reference_grid, x_bounds, y_bounds, x_width, y_width = utils.generate_reference_grid(config)
    
##### --- Put Hyperparameter into Config ---
    config['EM_MAX_ITER'] = args.em_max_iter
    config['ROUND'] = args.round
    
    config['X_BOUNDS'] = x_bounds
    config['Y_BOUNDS'] = y_bounds
    config['X_WIDTH'] = x_width
    config['Y_WIDTH'] = y_width

##### --- Create Checkpoint Directory and Path ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    SCENE_NAME = config.get('SCENARIO_NAME', 'Untitled_Scene')
    MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{SCENE_NAME}.ckpt")

##### --- Static Resource Loading and Initialization ---
    try:
        mat = scipy.io.loadmat("directions.mat")
        DIRECTIONS_VECTORS = mat['directions']

    except (FileNotFoundError, KeyError) as e:
        print(f"[TRANSFORMER ERROR] Could not load resource: {e}. Check if 'directions.mat' is in root and key is 'directions'. Exiting.")
        return

    device = DEVICE

##### --- Create 3 Queue and KeyboardInterrupt Stop Event ---
    csi_data_queue = Queue() # CSI Blocks Producter -> CSI2TRAJECTORY Worker
    trajectory_queue = Queue()  # CSI2TRAJECTORY Worker -> TRANSFORMER Worker
    model_queue = Queue() # TRANSFORMER Worker -> CSI2TRAJECTORY Worker
    stop_event = Event()

##### --- Load and Process CSI dataset ---
    csi_blocks_list = utils.load_and_preprocess_csi_dataset(
        dataset_folder=DATASET_FOLDER,
        config=config,
        device=device
    )

    if csi_blocks_list:
        for csi_block in csi_blocks_list:
            csi_data_queue.put(csi_block)
        print(f"\n[Pipeline] Queued {len(csi_blocks_list)} CSI blocks for processing.")
    else:
        print("[Pipeline] No CSI blocks generated. Exiting.")
        return

##### --- Processes for the Two-stage Pipeline
    csi2trajectory_worker_process = Process(
        target=CSI2TRAJECTORY_worker, 
        args=(csi_data_queue, trajectory_queue, model_queue, config, reference_grid, DIRECTIONS_VECTORS, device)
    )
    transformer_worker_process = Process(
        target=TRANSFORMER_worker, 
        args=(trajectory_queue, model_queue, config, DIRECTIONS_VECTORS, MODEL_CHECKPOINT_PATH, stop_event, device)
    )

##### --- Start the Concurrent Execution of the Two Worker Processes
    csi2trajectory_worker_process.start()
    transformer_worker_process.start()

##### --- # Wait for Workers to Complete, ensuring graceful termination and cleanup on interrupt
    try:
        csi2trajectory_worker_process.join()
        transformer_worker_process.join()
        print("Pipeline finished normally.")
    except KeyboardInterrupt:
        print("\n[Main] Terminating pipeline...")
        # Let Transformer Worker Save Current Model to Checkpoint
        stop_event.set()
        transformer_worker_process.join(timeout=10)

        if transformer_worker_process.is_alive():
            transformer_worker_process.terminate()
    finally:
        if csi2trajectory_worker_process.is_alive():
            csi2trajectory_worker_process.terminate()
            csi2trajectory_worker_process.join()
        print("[Main] Cleanup complete.")


##############################################
########## --- CSI to Trajectory --- #########
##############################################

def CSI2TRAJECTORY_worker(
    csi_data_queue: Queue,
    trajectory_queue: Queue, 
    model_queue: Queue, 
    config: Dict[str, Any], 
    reference_grid: Any,
    directions_vectors: np.ndarray, 
    device: torch.device
):

##### --- Initialize Model ---
    current_model_config = {
        'type': 'MARKOV',
    }
    current_transformer_model = None

##### --- LOS/NLOS ration for each AP --- 
    ap_data = config.get('ACCESS_POINTS', {})
    Q = len(ap_data)
    T = config['NUM_SAMPLE']
    SLEEP = config.get('CSItoTRAJECTORY_SLEEP_S')

    epsilon = 1e-6
    random_noise = torch.rand(Q, T, 2, device=device) * epsilon

##### --- Dynamic parameters ---
    context = {
        'last_predicted_point': None,
    }

    print("[CSI2TRAJ] Waiting for initial Transformer model state from Worker...")
    try:
        new_model_config = model_queue.get(timeout=3) # Wait for 3 Seconds

        if new_model_config['type'] == 'TRANSFORMER':
            current_transformer_model = create_transformer_instance(config, N_DIRECTIONS, device)
            current_transformer_model.load_state_dict(new_model_config['weights'])
            current_transformer_model.eval()
            current_model_config = new_model_config
            print(f"[CSI2TRAJ] Received initial model V{current_model_config.get('version', 'N/A')}. Starting trajectory generation in TRANSFORMER mode.")
        else:
            print("[CSI2TRAJ] Initial model setup is MARKOV or failed. Defaulting to MARKOV mode.")
            
    except Exception as e:
        print(f"[CSI2TRAJ] Error or timeout waiting for initial model: {e}. Defaulting to MARKOV mode.")

##### --- Implement CSItoTRAJ ---
    round_counter = 0
    try:
        while round_counter < config['ROUND']: # use while when it is RT system

            if not csi_data_queue.empty():  
                current_csi_block = csi_data_queue.get()
            else:
                continue

            # Check Whether There is a New Transformer Model can be Used
            if not model_queue.empty():
                new_model_config = model_queue.get()

                if new_model_config['type'] == 'TRANSFORMER':
                    if current_transformer_model is None:
                        current_transformer_model = create_transformer_instance(config, N_DIRECTIONS, device)

                    current_transformer_model.load_state_dict(new_model_config['weights'])
                    current_transformer_model.eval()
                    current_model_config = new_model_config
            else:
                pass

            if current_model_config['type'] == 'MARKOV':
                mode='MARKOV'
                model=None

            elif current_model_config['type'] == 'TRANSFORMER':
                mode='TRANSFORMER'
                model=current_transformer_model

            trajectory = (
                run_csi2traj(
                    config=config, 
                    reference_grid=reference_grid, 
                    context=context, 
                    model=model, 
                    mode=mode, 
                    directions_vectors=directions_vectors, 
                    raw_csi_data=current_csi_block,
                )
            )

            context['last_predicted_point'] = trajectory[-1:].clone().detach()

            trajectory_queue.put(trajectory.clone().detach())

            round_counter += 1
            time.sleep(SLEEP)

            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        return


##############################################
######## --- Transformer Training --- ########
##############################################

def TRANSFORMER_worker(
    trajectory_queue: Queue, 
    model_queue: Queue, 
    config: Dict[str, Any], 
    directions_vectors: np.ndarray, 
    checkpoint_path: str, 
    stop_event: Event,  # pyright: ignore[reportInvalidTypeForm]
    device: torch.device
):
##### --- Parameters Setup ---
    TRAINING_EPOCHS = config['TRAINING_EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    MIN_TRAJECTORIES_TO_TRAIN = config['MIN_TRAJ_TO_TRAIN']
    SLEEP = config['TRANSFORMER_SLEEP_S']

    SOS_TOKEN = N_DIRECTIONS

##### --- Instantiation ---
    model = create_transformer_instance(config, N_DIRECTIONS, device)
    
##### --- Optimization ---
    optim = NoamOpt(
        config['EMB_SIZE'],
        config['NOAMOPT_FACTOR'],
        config['NOAMOPT_WARMUP_STEPS'], 
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    trajectory_buffer: List[torch.Tensor] = []
    current_version = 0

##### --- Conditional Checkpoint Load ---

    checkpoint_loaded_successfully = False
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optim.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            loaded_step = checkpoint.get('noam_step', 0)
            if loaded_step > 0:
                optim._step = loaded_step

            current_version = checkpoint.get('version', 0)

            checkpoint_loaded_successfully = True
            
            print(f"[TRANSFORMER] Loaded checkpoint V{current_version} for scene '{os.path.basename(checkpoint_path)}'. Continuing training.")
            
        except Exception as e:
            print(f"[TRANSFORMER ERROR] Failed to load checkpoint {checkpoint_path}. Starting from scratch. Error: {e}")
    else:
        print(f"[TRANSFORMER] No checkpoint found. Starting V1 training from scratch.")

    if checkpoint_loaded_successfully:
        # Push Initial Model into Model Queue
            initial_model_config = {
                'version': current_version,
                'type': 'TRANSFORMER',
                'weights': model.state_dict(),
            }
            model_queue.put(initial_model_config)
    else:
        print("[TRANSFORMER] Cold Start or load failed. Skipping initial model push.")

##### --- Online Learning Core Loop ---
    try:
        while not stop_event.is_set():
            # Data Collection
            if not trajectory_queue.empty():
                trajectory_buffer.append(trajectory_queue.get())
                
            # To Trigger Training
            if len(trajectory_buffer) >= MIN_TRAJECTORIES_TO_TRAIN:
                print("[TRANSFORMER] Start Training...")

                model.train()
                trajs_for_batch = trajectory_buffer[:MIN_TRAJECTORIES_TO_TRAIN]
                data_tensor = torch.stack(trajs_for_batch, dim=0)
                
                try:
                    # inp_coords: (N, T-1, 2) | target_ids: (N, T-1)
                    inp_coords, target_ids = convert_long_trajectory_to_ids(
                        data_tensor, directions_vectors, device
                    )
                except Exception as e:
                    print(f"[TRANSFORMER ERROR] Long sequence conversion failed: {e}. Skipping batch.")
                    import traceback
                    traceback.print_exc()
                    trajectory_buffer = trajectory_buffer[MIN_TRAJECTORIES_TO_TRAIN:]
                    continue
                    
                train_dataset = TensorDataset(inp_coords, target_ids)

                total_batches = 0
                current_epoch_loss = 0.0

                for epoch_i in range(TRAINING_EPOCHS):
                    tr_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                    epoch_loss = 0.0

                    for id_b, (inp, target) in enumerate(tr_dl):
                        
                        optim.optimizer.zero_grad()
                        n_in_batch = inp.shape[0]
                        
                        # Prepare the Causal Mask
                        trg_att = subsequent_mask(target.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                        
                        # Prepare Teacher Forcing input: [SOS | Target IDs (x2...xT-1)]
                        start_of_seq = torch.tensor([SOS_TOKEN]).repeat(n_in_batch).unsqueeze(1).to(device)
                        dec_inp = torch.cat((start_of_seq, target[:,:-1]), 1)
                        
                        # Encoder(inp) + Decoder(dec_inp_teacher)
                        out = model(inp, dec_inp, None, trg_att)
                        
                        # Loss Calculation (Cross-Entropy)
                        loss = F.cross_entropy(out.view(-1, out.shape[-1]), target.view(-1), reduction='mean')
                        
                        loss.backward()
                        optim.step()

                        time.sleep(SLEEP)
                        epoch_loss += loss.item()
                        total_batches += 1
                        current_epoch_loss += loss.item()
                    
                    # print(f"[TRANSFORMER] V{current_version + 1} Epoch {epoch_i + 1}/{TRAINING_EPOCHS} Loss: {epoch_loss/len(tr_dl):.4f}")
                    
                # Model Publication
                current_version += 1
                avg_loss = current_epoch_loss / total_batches
                new_model_config = {
                    'version': current_version,
                    'type': 'TRANSFORMER',
                    'weights': model.state_dict(),
                }
                
                model_queue.put(new_model_config)
                print(f"[TRANSFORMER] Finished training V{current_version}. Final Avg Loss: {avg_loss:.4f}. Sent to CSI Worker.")
                
                # Clean up the used buffer
                trajectory_buffer = trajectory_buffer[MIN_TRAJECTORIES_TO_TRAIN:]

                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                
            else:
                time.sleep(SLEEP)

    except KeyboardInterrupt:
        print(f"\n[TRANSFORMER] KeyboardInterrupt received. Setting stop_event for graceful checkpoint save.")
        stop_event.set()

    # Final state save
    print(f"[TRANSFORMER] Saving final state V{current_version} before exiting...")
    try:
        torch.save({
            'version': current_version,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.optimizer.state_dict(),
            'noam_step': optim._step,
        }, checkpoint_path)
        print(f"[TRANSFORMER] Successfully saved final state V{current_version}.")
    except Exception as e:
        print(f"[TRANSFORMER ERROR] Failed to save checkpoint at {checkpoint_path}. Error: {e}")


if __name__ == '__main__':
    main()