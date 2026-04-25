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
    parser = argparse.ArgumentParser(description="CSI Indoor Localization System")

    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--em-max-iter", dest="em_max_iter", type=int, default=None)

    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--enable-trajectory-decoding", action="store_true", default=None)
    parser.add_argument("--enable-neural", action="store_true", default=None)

    parser.add_argument("--csi-datasets", nargs="+", default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)

    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--updates-per-batch", type=int, default=None)

    parser.add_argument("--publish-interval", type=int, default=None)
    parser.add_argument("--min-updates-before-publish", type=int, default=None)
    args = parser.parse_args()

    # 3. Load Configuration
    config = utils.load_yaml_config(args.config)
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
    if args.em_max_iter is not None: config["EM_MAX_ITER"] = args.em_max_iter

    if args.method is not None: config["METHOD"] = args.method
    if args.enable_neural: config["ENABLE_NEURAL"] = True
    if args.enable_trajectory_decoding: config["ENABLE_TRAJECTORY_DECODING"] = True

    if args.csi_datasets is not None: config["CSI_DATASETS"] = args.csi_datasets

    if args.output_dir is not None: config["OUTPUT_DIR"] = args.output_dir
    if args.checkpoint_dir is not None: config["CHECKPOINT_DIR"] = args.checkpoint_dir
    output_dir = config.get("OUTPUT_DIR", OUTPUT_DIR)
    checkpoint_dir = config.get("CHECKPOINT_DIR", CHECKPOINT_DIR)

    if args.batch_size is not None: config["NEURAL_BATCH_SIZE"] = args.batch_size
    if args.window_size is not None: config["NEURAL_WINDOW_SIZE"] = args.window_size
    if args.window_stride is not None: config["NEURAL_WINDOW_STRIDE"] = args.window_stride

    if args.dropout is not None: config["NEURAL_DROPOUT"] = args.dropout
    if args.learning_rate is not None: config["LEARNING_RATE"] = args.learning_rate
    if args.updates_per_batch is not None: config["NEURAL_UPDATES_PER_BATCH"] = args.updates_per_batch

    if args.publish_interval is not None: config["NEURAL_PUBLISH_INTERVAL"] = args.publish_interval
    if args.min_updates_before_publish is not None: config["NEURAL_MIN_UPDATES_BEFORE_PUBLISH"] = args.min_updates_before_publish

    # Symbolic is always required; neural is optional
    enable_neural = config.get('ENABLE_NEURAL', True)
    
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
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "grid.npy"), grid_np)

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
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    csi_datasets = config.get('CSI_DATASETS', [])

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
    print(f"[System] Initializing workers in {config.get('METHOD')} mode...")

    symbolic_worker = SymbolicWorker(
        name="SymbolicWorker",
        config=config,
        queues=queues,
        stop_event=stop_event,
        reference_grid=reference_grid,
        directions_vectors=directions_vectors
    )

    if enable_neural:
        neural_worker = NeuralWorker(
            name="NeuralWorker",
            config=config,
            queues=queues,
            stop_event=stop_event
        )

    # 7. Start Processes
    symbolic_worker.start()
    if enable_neural:
        neural_worker.start()

    # 8. Main
    try:
        print(f"\n{'='*50}")
        print("[System] Starting CSI dataset processing")
        print(f"{'='*50}")

        for csi_dataset in csi_datasets:
            print(f"\n{'='*50}")
            print(f"[System] Processing CSI dataset: {csi_dataset}")
            print(f"{'='*50}")

            hmatrix_folder = os.path.join(dataset_folder, csi_dataset)
            csi_blocks = utils.get_csi_blocks_with_cache(
                hmatrix_folder=hmatrix_folder,
                config=config
            )

            if not csi_blocks:
                print(f"[System] No data in {csi_dataset}.")
                continue

            total_batches = len(csi_blocks)
            print(f"[System] Injecting {total_batches} blocks from {csi_dataset}...")

            for block in csi_blocks:
                queues["data_symbolic"].put(block.cpu())

            if enable_neural:
                queues["data_symbolic"].put({
                    "type": "end_of_sequence",
                    "source": csi_dataset,
                    "num_blocks": total_batches,
                })

            print(f"[System] Waiting for {total_batches} results from {csi_dataset}...")

            dataset_trajectory = []
            dataset_emission_log_probs = []
            dataset_posterior = []
            dataset_reliability = []
            dataset_aggregated_csi = []

            received_count = 0
            while received_count < total_batches:
                payload = queues["out"].get(timeout=None)
                received_count += 1

                trajectory = utils.to_numpy(payload.get("trajectory"))
                if trajectory is not None:
                    dataset_trajectory.append(trajectory)

                emission_log_probs_qgt = utils.to_numpy(payload.get("emission_log_probs_qgt"))
                if emission_log_probs_qgt is not None:
                    dataset_emission_log_probs.append(emission_log_probs_qgt)

                posterior_gt = utils.to_numpy(payload.get("posterior_gt"))
                if posterior_gt is not None:
                    dataset_posterior.append(posterior_gt)

                reliability = utils.to_numpy(payload.get("reliability"))
                if reliability is not None:
                    dataset_reliability.append(reliability)

                aggregated_csi = utils.to_numpy(payload.get("aggregated_csi"))
                if aggregated_csi is not None:
                    dataset_aggregated_csi.append(aggregated_csi)

            queues["data_symbolic"].join()
            if enable_neural:
                queues["data_neural"].join()

            dataset_output_dir = os.path.join(output_dir, csi_dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)

            print(f"[System] Saving outputs to {dataset_output_dir}...")

            if dataset_trajectory:
                np.save(
                    os.path.join(dataset_output_dir, "predicted_trajectory.npy"),
                    np.concatenate(dataset_trajectory, axis=0)
                )

            if dataset_emission_log_probs:
                np.save(
                    os.path.join(dataset_output_dir, "emission_log_probs_qgt.npy"),
                    np.concatenate(dataset_emission_log_probs, axis=2)
                )

            if dataset_posterior:
                np.save(
                    os.path.join(dataset_output_dir, "posterior_gt.npy"),
                    np.concatenate(dataset_posterior, axis=1)
                )

            if dataset_reliability:
                np.save(
                    os.path.join(dataset_output_dir, "reliability.npy"),
                    np.stack(dataset_reliability, axis=0)
                )

            if dataset_aggregated_csi:
                np.save(
                    os.path.join(dataset_output_dir, "aggregated_csi.npy"),
                    np.stack(dataset_aggregated_csi, axis=0)
                )

            print(f"[System] Finished CSI dataset: {csi_dataset}")

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    
    finally:
        # 9. Graceful Shutdown
        print("[System] Shutting down workers...")
        stop_event.set()

        while not queues['out'].empty(): queues['out'].get()
        
        symbolic_worker.join(timeout=2)
        if symbolic_worker.is_alive(): symbolic_worker.terminate()
        if enable_neural:
            neural_worker.join(timeout=2)
            if neural_worker.is_alive(): neural_worker.terminate()
        
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()