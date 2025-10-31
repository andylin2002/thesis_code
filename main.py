import utils
import torch
import argparse
from typing import List, Dict, Any
import time
from multiprocessing import Queue, Process, set_start_method

from csi2traj import run_csi2traj
#from transformer.trainer import create_transformer_instance (FIXME)

CONFIG_PATH = 'config.yaml'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--round', type=int, default=5)

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

    device = DEVICE

    data_queue = Queue()  # CSI2TRAJECTORY Worker -> TRANSFORMER Worker
    model_queue = Queue() # TRANSFORMER Worker -> CSI2TRAJECTORY Worker

##### --- Processes for the Two-stage Pipeline
    csi2trajectory_worker_process = Process(
        target=CSI2TRAJECTORY_worker, 
        args=(data_queue, model_queue, config, reference_grid, device)
    )
    transformer_worker_process = Process(
        target=TRANSFORMER_worker, 
        args=(data_queue, model_queue, config, device)
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
                    current_transformer_model = create_transformer_instance(config, device)

                current_transformer_model.load_state_dict(new_model_config['weights'])
                current_transformer_model.eval()
                current_model_config = new_model_config
        else:
            pass

        if current_model_config['type'] == 'MARKOV':
            trajectory = (
                run_csi2traj(
                    config=config, 
                    reference_grid=reference_grid, 
                    context=context, 
                    model=None, 
                    mode='MARKOV'
                )
            )
        elif current_model_config['type'] == 'TRANSFORMER':
            trajectory = (
                run_csi2traj(
                    config=config, 
                    reference_grid=reference_grid, 
                    context=context, 
                    model=current_transformer_model, 
                    mode='TRANSFORMER'
                )
            )

        context['last_predicted_point'] = trajectory[-1:].clone().detach()

        data_queue.put(trajectory.clone().detach())

        round_counter += 1
        time.sleep(0.1) # (FIXME: 用arg或config來調整)


##############################################
######## --- Transformer Training --- ########
##############################################

def TRANSFORMER_worker(
    data_queue: Queue, 
    model_queue: Queue, 
    config: Dict[str, Any], 
    device: torch.device
):
    try:
        # 讓 Worker 保持運行，但消耗極少的資源，等待 CSI Worker 結束
        while True:
            # 可選：從隊列中取出數據，防止隊列溢出，但不進行訓練
            if not data_queue.empty():
                _ = data_queue.get()
            time.sleep(1) 
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()