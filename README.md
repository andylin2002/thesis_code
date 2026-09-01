# Neuro–Symbolic Wi-Fi CSI Indoor Localization

Code accompanying the master's thesis **Neuro–Symbolic Wi-Fi CSI Indoor Localization via Multipath-Aware Trajectory Recovery and Reliability Modulation** by Tzu-Chuan Lin, National Cheng Kung University, 2026.

The code supports baseline and proposed symbolic localization, pretrained AP reliability inference, and optional online neural training.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/andylin2002/Neuro--Symbolic_Wi-Fi_CSI_Indoor_Localization.git
cd Neuro--Symbolic_Wi-Fi_CSI_Indoor_Localization
```

### 2. Set up Python

The current implementation requires a **CUDA-capable PyTorch environment**.

```bash
python -m venv venv
source venv/bin/activate
```

Install a CUDA-enabled PyTorch build suitable for your system, then install the remaining dependencies:

```bash
python -m pip install numpy scipy pyyaml pandas matplotlib scikit-learn
```

### 3. Download the dataset

The dataset is not stored in the Git repository.

Download the Release asset named `dataset.tar.gz` from the repository's [Releases](https://github.com/andylin2002/thesis_code/releases) page and place it in the repository root.

Extract it with:

```bash
tar -xzf dataset.tar.gz
```

This creates:

```text
dataset/
├── dataset_easyspace/
├── dataset_office/
└── dataset_office_8AP/
```

### 4. Select a CSI sequence

The default configuration uses `dataset/dataset_office_8AP/`.

Edit `CSI_DATASETS` in `config.yaml`, for example:

```yaml
CSI_DATASETS:
  - "train_wander"
```

### 5. Run

```bash
python main.py
```

The default configuration runs the proposed method with online neural training disabled.

If `checkpoint/Office_8AP.ckpt` is available, pretrained reliability inference is still used. See [Usage](#usage) for the available execution modes.

## Repository Structure

The tree below shows the core runtime code and offline analysis utilities. Large datasets, experiment outputs, virtual environments, caches, and generated files are omitted.

```text
.
├── main.py
├── config.yaml
├── directions.mat
├── utils.py
│
├── analysis/                       # Offline analysis and parameter-search utilities
│   ├── analysis_config.yaml
│   ├── run_analysis.py
│   ├── run_symbolic_ablation.py
│   ├── run_neural_param_search.py
│   ├── run_neural_seed_search.py
│   └── ...
│
├── core/
│   ├── interfaces.py
│   └── models/                     # Neural reliability model
│
├── engines/
│   ├── symbolic_engine/
│   │   ├── runtime.py
│   │   ├── modules/                # Component factory
│   │   ├── strategies/             # Baseline / proposed implementations
│   │   └── stages/
│   │       ├── signal_processing/  # CSI processing and feature extraction
│   │       ├── result_estimation/  # HardEM / SoftEM / Viterbi
│   │       └── gating_evaluation/  # Pretrained reliability inference
│   │
│   └── neural_engine/
│       ├── runtime.py
│       └── stages/
│           ├── load.py             # Block accumulation and windows
│           ├── represent.py        # Neural CSI representation
│           ├── proxy.py            # Training target construction
│           └── train.py            # Neural optimization
│
├── workers/
│   ├── symbolic_worker.py
│   └── neural_worker.py
│
└── checkpoint/
    └── Office_8AP.ckpt
```

### Main entry points

| Path | Purpose |
| --- | --- |
| `main.py` | Loads configuration and datasets, starts workers, and saves outputs |
| `config.yaml` | Main runtime and algorithm configuration |
| `analysis/` | Optional offline ablation, parameter-search, seed-search, and analysis utilities |
| `workers/symbolic_worker.py` | Foreground CSI localization process |
| `workers/neural_worker.py` | Optional background neural-training process |
| `engines/symbolic_engine/` | Symbolic signal processing and localization |
| `engines/neural_engine/` | Neural reliability training |
| `core/models/` | Neural reliability network |

For code reading, start from `main.py`, then follow the workers into the corresponding engine runtime and stages.

The `analysis/` directory is not required for standard localization runs.

## Dataset

After extracting `dataset.tar.gz`, the repository should contain:

```text
dataset/
├── dataset_easyspace/
├── dataset_office/
└── dataset_office_8AP/
```

The default configuration selects:

```yaml
DATASET_FOLDER: "dataset_office_8AP"
ENV_CONFIG: "env_config.yaml"
```

The selected directory contains:

```text
dataset/dataset_office_8AP/
├── env_config.yaml
├── ground_truth/
├── LoS_t150_ch106_ant3/
├── LoS_t150_ch106_ant8/
├── LoS_t150_ch36_ant3/
├── NLoS_t150_ch106_ant3/
├── NLoS_t150_ch106_ant8/
├── NLoS_t150_ch36_ant3/
├── NLoS_t799_ch36_ant3/
├── train_circle/
├── train_direct/
├── train_long/
├── train_wander/
├── dataset_ML/
└── UWB/
```

Each value in `CSI_DATASETS` is interpreted relative to the selected dataset folder. For example:

```yaml
CSI_DATASETS:
  - "train_wander"
```

loads:

```text
dataset/dataset_office_8AP/train_wander/
```

### Custom CSI data

The CSI loader accepts Wireless InSite-style filenames matching:

```text
t<timestamp>_hmatrix.txSet<id>.txPt<id>.rxSet<AP>.inst<subcarrier>.csv
```

Example:

```text
t1000_hmatrix.txSet014.txPt001.rxSet001.inst001.csv
```

A sequence may also contain `cache.npy`. If present, the cache is loaded instead of reparsing the CSV files. Otherwise, the cache is created automatically.

Delete or regenerate `cache.npy` when the antenna or subcarrier dimensions change.

## Configuration

Configuration is applied in the following order:

```text
config.yaml
    ↓
dataset/<DATASET_FOLDER>/<ENV_CONFIG>
    ↓
command-line overrides
```

Later values override earlier ones.

### Main settings

```yaml
METHOD: "PROPOSED"              # BASELINE or PROPOSED
ENABLE_TRAJECTORY_DECODING: true
ENABLE_NEURAL_TRAINING: false

OUTPUT_DIR: "output"
CHECKPOINT_DIR: "checkpoint"

DATASET_FOLDER: "dataset_office_8AP"
ENV_CONFIG: "env_config.yaml"

CSI_DATASETS:
  - "train_wander"
```

### CSI settings

```yaml
GRID_RESOLUTION_M: 0.5

CSI_DIMENSIONS:
  NUM_RX_ANTENNAS: 3
  NUM_SUBCARRIERS: 21

CARRIER_FREQUENCY_HZ: 5180000000.0
CHANNEL_BANDWIDTH_HZ: 20000000.0
ANTENNA_DISTANCE: 0.02899

NUM_SAMPLE: 15
NUM_PACKET: 1
```

### Symbolic settings

```yaml
EM_MAX_ITER: 100

BASELINE_AOA_METHOD: "mmp"      # mmp or music

ENABLE_SOFT_EM: true
ENABLE_TOF_GAIN_WEIGHT: true
ENABLE_MULTIPATH: true
```

### Neural settings

```yaml
NEURAL_BATCH_SIZE: 64
NEURAL_WINDOW_SIZE: 16
NEURAL_WINDOW_STRIDE: 4
NEURAL_UPDATES_PER_BATCH: 100

NEURAL_PUBLISH_INTERVAL: 1
NEURAL_MIN_UPDATES_BEFORE_PUBLISH: 1

LEARNING_RATE: 0.004
WEIGHT_DECAY: 0.0001
GRADIENT_CLIP_NORM: 1.0

NEURAL_DROPOUT: 0.2
```

For the complete configuration, see `config.yaml`.

## Usage

| Mode | Method | Checkpoint | Online neural training |
| --- | --- | --- | --- |
| Baseline | `BASELINE` | Not used for gating | No |
| Proposed symbolic only | `PROPOSED` | Empty / unavailable | No |
| Proposed + pretrained reliability | `PROPOSED` | Existing | No |
| Proposed + online neural training | `PROPOSED` | Existing or new | Yes |

### Baseline

```bash
python main.py \
    --method BASELINE \
    --csi-datasets <dataset_name>
```

### Proposed symbolic only

`ENABLE_NEURAL_TRAINING=false` disables online training, but it does not disable pretrained reliability inference from an existing checkpoint.

To guarantee a symbolic-only run:

```bash
mkdir -p checkpoint_symbolic_only
rm -f checkpoint_symbolic_only/Office_8AP.ckpt

python main.py \
    --method PROPOSED \
    --checkpoint-dir checkpoint_symbolic_only \
    --csi-datasets <dataset_name>
```

Keep:

```yaml
ENABLE_NEURAL_TRAINING: false
```

### Proposed with pretrained reliability

Using `checkpoint/Office_8AP.ckpt`:

```bash
python main.py \
    --method PROPOSED \
    --checkpoint-dir checkpoint \
    --csi-datasets <dataset_name>
```

with:

```yaml
ENABLE_NEURAL_TRAINING: false
```

The checkpoint is used for reliability inference but is not updated.

### Proposed with online neural training

Resume training from the supplied checkpoint:

```bash
python main.py \
    --method PROPOSED \
    --enable-neural-training \
    --checkpoint-dir checkpoint \
    --csi-datasets <dataset_name>
```

Start from a new checkpoint directory:

```bash
mkdir -p checkpoint_scratch
rm -f checkpoint_scratch/Office_8AP.ckpt

python main.py \
    --method PROPOSED \
    --enable-neural-training \
    --checkpoint-dir checkpoint_scratch \
    --csi-datasets <dataset_name>
```

For from-scratch training, provide enough input blocks to complete at least one `NEURAL_BATCH_SIZE` batch before treating the saved checkpoint as a trained model.

### Multiple sequences

```bash
python main.py \
    --csi-datasets train_direct train_circle train_wander
```

### Command-line options

Run:

```bash
python main.py --help
```

for the complete list of available overrides.

## Outputs

Results for each processed sequence are written to:

```text
<OUTPUT_DIR>/<dataset_name>/
```

Typical proposed-method outputs are:

| File | Description | Typical shape |
| --- | --- | --- |
| `trajectory.npy` | Decoded coordinates | `[total_time_steps, 2]` |
| `aggregated_csi.npy` | Aggregated CSI | `[num_blocks, Q, T, N, M]` |
| `emission_log_probs_qgt.npy` | AP-wise symbolic log-emissions | `[Q, G, total_time_steps]` |
| `posterior_gt.npy` | Forward–Backward grid posterior | `[G, total_time_steps]` |
| `reliability.npy` | Available AP-time reliability values | `[num_available_blocks, Q, T]` |
| `timing_report.json` | Runtime and throughput statistics | JSON |

`reliability.npy` is created from blocks for which reliability values are available, so its first dimension may be smaller than the number of processed CSI blocks.

## Notes

- The current implementation requires CUDA in practice because reference-grid generation uses CUDA directly.
- `ENABLE_NEURAL_TRAINING=false` does not disable pretrained reliability inference from an existing checkpoint.
- Online neural training is safest with `OUTPUT_DIR: "output"` because the current symbolic and neural code paths expect `neighbor_matrix.npy` at that location.
- `directions.mat` must exist in the repository root and contain a MATLAB variable named `directions`.
- Dependency versions are not currently pinned.

## Citation

```bibtex
@mastersthesis{lin2026neurosymbolic,
  author = {Tzu-Chuan Lin},
  title  = {Neuro--Symbolic Wi-Fi CSI Indoor Localization via Multipath-Aware Trajectory Recovery and Reliability Modulation},
  school = {National Cheng Kung University},
  year   = {2026},
  month  = {July}
}
```