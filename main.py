import utils
import torch
import argparse
from typing import List, Dict, Any
import time
from multiprocessing import Queue, Process, set_start_method
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F, numpy as np, scipy.io, os

from csi2traj import run_csi2traj
from transformer.transformer_tool import convert_long_trajectory_to_ids, create_transformer_instance
from transformer.architecture.noam_opt import NoamOpt
from transformer.architecture.batch import subsequent_mask

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
    parser.add_argument('--em_max_iter', type=int, default=30)
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

##### --- Static Resource Loading and Initialization ---
    try:
        mat = scipy.io.loadmat("directions.mat")
        DIRECTIONS_VECTORS = mat['directions']

    except (FileNotFoundError, KeyError) as e:
        print(f"[Trainer ERROR] Could not load resource: {e}. Check if 'directions.mat' is in root and key is 'directions'. Exiting.")
        return

    device = DEVICE

    data_queue = Queue()  # CSI2TRAJECTORY Worker -> TRANSFORMER Worker
    model_queue = Queue() # TRANSFORMER Worker -> CSI2TRAJECTORY Worker

##### --- Processes for the Two-stage Pipeline
    csi2trajectory_worker_process = Process(
        target=CSI2TRAJECTORY_worker, 
        args=(data_queue, model_queue, config, reference_grid, DIRECTIONS_VECTORS, device)
    )
    transformer_worker_process = Process(
        target=TRANSFORMER_worker, 
        args=(data_queue, model_queue, config, DIRECTIONS_VECTORS, device)
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
    finally:
        csi2trajectory_worker_process.terminate()
        transformer_worker_process.terminate()
        csi2trajectory_worker_process.join()
        transformer_worker_process.join()


##############################################
########## --- CSI to Trajectory --- #########
##############################################

def CSI2TRAJECTORY_worker(
    data_queue: Queue, 
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

    APs_LOS_ratio_symmetric = torch.full((Q, T, 2), 0.5, dtype=torch.float32, device=device)
    APs_LOS_ratio = APs_LOS_ratio_symmetric + random_noise

##### --- Dynamic parameters ---
    context = {
        'last_predicted_point': None,
        'APs_LOS_ratio': APs_LOS_ratio
    }

##### --- Implement CSItoTRAJ ---
    round_counter = 0
    while round_counter < config['ROUND']: # use while when it is RT system

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
                directions_vectors=directions_vectors
            )
        )

        context['last_predicted_point'] = trajectory[-1:].clone().detach()

        data_queue.put(trajectory.clone().detach())

        round_counter += 1
        time.sleep(SLEEP)


##############################################
######## --- Transformer Training --- ########
##############################################

def TRANSFORMER_worker(
    data_queue: Queue, 
    model_queue: Queue, 
    config: Dict[str, Any], 
    directions_vectors: np.ndarray, 
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

##### --- Online Learning Core Loop ---
    while True:
        # Data Collection
        if not data_queue.empty():
            trajectory_buffer.append(data_queue.get())
            
        # To Trigger Training
        if len(trajectory_buffer) >= MIN_TRAJECTORIES_TO_TRAIN:
            print("[TRANSFORMER] Start Training...")

            model.train()
            trajs_for_batch = trajectory_buffer[:MIN_TRAJECTORIES_TO_TRAIN]
            data_tensor = torch.stack(trajs_for_batch, dim=0) 
            
            try:
                # inp_coords: (N, T-1, 2) | target_ids: (N, T-1)
                inp_coords, target_ids = convert_long_trajectory_to_ids(
                    data_tensor, directions_vectors, config, device
                )
            except Exception as e:
                print(f"[Trainer ERROR] Long sequence conversion failed: {e}. Skipping batch.")
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
            
        else:
            time.sleep(SLEEP)


if __name__ == '__main__':
    main()