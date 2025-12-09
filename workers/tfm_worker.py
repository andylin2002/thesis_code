import torch
import os
import time
import numpy as np
from core.interfaces import BaseWorker
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import Transformer utilities
from transformer.transformer_tool import convert_long_trajectory_to_ids, create_transformer_instance
from transformer.architecture.noam_opt import NoamOpt
from transformer.architecture.batch import subsequent_mask

class TFM_Worker(BaseWorker):
    """
    Worker for online training. 
    Accumulates data from CSI Worker, trains the model, and publishes updates.
    """

    def __init__(self, name, config, queues, stop_event, directions_vectors, checkpoint_path):
        super().__init__(name, config, queues, stop_event)
        self.directions_vectors = directions_vectors
        self.checkpoint_path = checkpoint_path
        
        # Hyperparameters
        self.min_traj_len = config.get('MIN_TRAJ_TO_TRAIN', 120) 
        self.epochs = config.get('TRAINING_EPOCHS', 1)
        self.batch_size = config.get('BATCH_SIZE', 32)
        
        self.model = None
        self.optimizer = None
        self.active = False 
        self.current_version = 0

    def _setup(self):
        """
        Initialize Model, Optimizer, load Checkpoint, and push initial weights.
        """
        # Check System Mode
        if self.config.get('SYSTEM_MODE') == 'BASELINE':
            print(f"[{self.name}] Baseline mode. Worker standing by.")
            self.active = False
            return
        
        self.active = True
        
        # Create Model Architecture
        n_directions = 9
        self.model = create_transformer_instance(self.config, n_directions, self.device)
        
        # Setup Optimizer
        self.optimizer = NoamOpt(
            self.config['EMB_SIZE'],
            self.config['NOAMOPT_FACTOR'],
            self.config['NOAMOPT_WARMUP_STEPS'],
            torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        )

        # Load Checkpoint
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optimizer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.current_version = ckpt.get('version', 0)
                print(f"[{self.name}] Loaded checkpoint V{self.current_version}")
            except Exception as e:
                print(f"[{self.name}] Checkpoint load failed: {e}. Starting fresh.")
        
        # Push Initial Model
        try:
            init_weights = {
                'version': self.current_version,
                'type': 'TRANSFORMER',
                'weights': {k: v.cpu() for k, v in self.model.state_dict().items()}
            }
            self.queues['model'].put(init_weights)
            print(f"[{self.name}] Pushed initial model V{self.current_version} to queue.")
        except Exception as e:
            print(f"[{self.name}] Failed to push initial model: {e}")

        # Initialize Accumulation Buffer
        self.train_buffer = [] 

    def _loop(self):
        if not self.active:
            self.stop_event.wait()
            return

        in_queue = self.queues['result'] # Recv from CSI Worker
        out_queue = self.queues['model'] # Send to CSI Worker

        while not self.stop_event.is_set():
            # Check queue not empty
            if in_queue.empty():
                time.sleep(0.1)
                continue

            # 1. Accumulate Data
            # pkg format: {'input': {features, spd, epd}, 'pseudo_gt': path}
            pkg = in_queue.get()
            self.train_buffer.append(pkg)

            # 2. Check Trigger Condition
            if len(self.train_buffer) >= self.min_traj_len:
                print(f"[{self.name}] Buffer full ({len(self.train_buffer)}). Training...")
                
                try:
                    # 3. Run Training
                    avg_loss = self._run_training_step()
                    
                    # 4. Publish New Weights
                    self.current_version += 1
                    new_weights = {
                        'version': self.current_version,
                        'type': 'TRANSFORMER',
                        'weights': {k: v.cpu() for k, v in self.model.state_dict().items()}
                    }
                    out_queue.put(new_weights)
                    print(f"[{self.name}] Published V{self.current_version} (Loss: {avg_loss:.4f})")
                    
                    # 5. Save Checkpoint
                    self._save_checkpoint()

                except Exception as e:
                    print(f"[{self.name}] Training Failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 6. Reset Buffer (Simple reset strategy)
                self.train_buffer = [] 

    def _run_training_step(self):
        self.model.train()
        
        # A. Prepare Data: Stack list of dicts into Batch Tensor
        # Result Shape: (Batch=1, Time, Dim)
        
        # Extract individual components
        # Note: Need to detach to prevent graph retention issues across loops
        features_list = [x['input']['features'].detach() for x in self.train_buffer]
        spd_list = [x['input']['spd'].detach() for x in self.train_buffer]
        # Handle cases where EPD might not exist (e.g. Baseline logic leakage)
        epd_list = [x['input'].get('epd', torch.zeros(1)).detach() for x in self.train_buffer] 
        gt_list = [x['pseudo_gt'].detach() for x in self.train_buffer]

        # Stack
        inp_feats = torch.stack(features_list, dim=0).unsqueeze(0).to(self.device)
        inp_spd = torch.stack(spd_list, dim=0).unsqueeze(0).to(self.device)
        # Ensure EPD dimensions are correct if used
        if len(epd_list[0].shape) > 0:
             inp_epd = torch.stack(epd_list, dim=0).unsqueeze(0).to(self.device)
        else:
             inp_epd = None # or handle accordingly in model

        gt_path = torch.stack(gt_list, dim=0).to(self.device) # (Time, 2)

        # B. Convert Path to Labels
        # We need (1, T-1) target IDs for CrossEntropy
        # Use helper tool to get deltas -> direction IDs
        _, target_ids = convert_long_trajectory_to_ids(
            gt_path.unsqueeze(0), self.directions_vectors, self.device
        )

        total_loss = 0
        
        # C. Training Epochs on this sequence
        for _ in range(self.epochs):
            self.optimizer.optimizer.zero_grad()
            
            # Causal Mask
            tgt_mask = subsequent_mask(target_ids.shape[1]).to(self.device)
            
            # Decoder Input (Teacher Forcing: Shift Right)
            SOS_TOKEN = 9 
            start_token = torch.tensor([[SOS_TOKEN]], device=self.device)
            dec_inp = torch.cat([start_token, target_ids[:, :-1]], dim=1)
            
            # Forward
            output = self.model(
                src_features=inp_feats, 
                src_spd=inp_spd, 
                src_epd=inp_epd, 
                tgt=dec_inp, 
                src_mask=None, 
                tgt_mask=tgt_mask
            )
            
            # Loss
            loss = F.cross_entropy(
                output.view(-1, output.shape[-1]), 
                target_ids.view(-1)
            )
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / self.epochs

    def _save_checkpoint(self):
        try:
            torch.save({
                'version': self.current_version,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.optimizer.state_dict()
            }, self.checkpoint_path)
        except Exception as e:
            print(f"[{self.name}] Failed to save checkpoint: {e}")