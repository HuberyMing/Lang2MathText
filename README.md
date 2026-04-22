# Lang2MathText

**fMRI Brain Decoding with LLM Embeddings**

This project investigates whether sentence embeddings from different layers of large language models (LLMs) can predict fMRI brain activation patterns in response to math vs. language stimuli.

---

## Research Question

Can LLM layer embeddings predict neural responses to sentences across math and language domains? The pipeline sweeps over all LLM layers and evaluates Pearson correlation between predicted and actual fMRI signals per brain region.

---

## Two-Stage Pipeline

| Stage | Script | Where to run | What it does |
|-------|--------|-------------|-------------|
| 1 | `src/run_LLM.py` | GPU server (H100) | Load LLM, extract per-layer embeddings for stimulus sentences, save to CSV |
| 2 | `src/run_nested_cv.py` | Mac (CPU OK) | Load pre-computed embeddings, run nested CV to predict fMRI responses, evaluate per-ROI Pearson correlation |

---

## File Structure

```
Lang2MathText/
│
├── config/                                 # LLM and data configurations
│   ├── data_config.yaml                    # fMRI dataset config (file path, stimulus mapping, conditions)
│   ├── FoxBrain_70B_SFT_100K_DPO.json     # FoxBrain 70B model config
│   ├── Llama-3.3-70B-Instruct.json        # Llama 70B model config
│   └── Qwen3-32B.json                     # Qwen3 32B model config
│
├── src/                                    # Source code (canonical version)
│   ├── run_nested_cv.py                   # ★ Stage 2 entry point — nested CV sweep over layers
│   ├── run_LLM.py                         # ★ Stage 1 entry point — LLM embedding extraction
│   ├── preprocessing.py                   # ScalePreprocessor: L2 norm → PCA → SelectKBest
│   ├── data_module.py                     # FMRIDataModule: data alignment & shuffle (no leakage)
│   ├── orchestrator.py                    # AnalysisOrchestrator, ExperimentRunner, run_nested_balanced_cv
│   ├── cv_utils.py                        # Balanced group CV splits, fold averaging
│   ├── model_adapters.py                  # Adapter layer: SklearnAdapter, PyTorchLightningAdapter
│   │
│   ├── data/
│   │   └── fMRI_data_loader.py            # Load_dataset(): load fMRI data from CSV
│   │
│   ├── LLMmodels/
│   │   └── embeddings_transf.py           # load_embeddings_csv(), run_llm_and_save(), mean_pooling()
│   │
│   └── utils/
│       ├── plotting.py                    # All plotting functions (scatter, layer curves, brain maps)
│       ├── metrics.py                     # calculate_voxel_correlation(), calculate_regression_metrics()
│       └── helper.py                      # load_config(): YAML config loader
│
├── data/                                   # ⚠ Not tracked by git (files too large)
│   ├── raw/                               # Raw fMRI CSV files and stimulus files
│   └── processed/                         # Pre-computed LLM embedding CSVs (output of Stage 1)
│                                          # Filename format: {model_name}_emb_L{layer}_{pooling}_utf8.csv
│
└── results/                               # ⚠ Not tracked by git (output of Stage 2)
    └── {model_name}/
        ├── ALL_layers/
        │   └── model_corr_layer_roi.json  # Per-layer × per-ROI Pearson correlation
        └── L{layer}_{pooling}/K{k}/
            └── scatters/                  # Scatter plots per ROI
```

> **Note:** `results/` directory name is set by `ParentDir` in `run_nested_cv.py` (currently `../results_Gemini_paper_2g6b`).

---

## Setup

```bash
# Create/activate conda environment
conda activate /Users/mervyn/miniconda3/envs/fMRI

# Run from src/
cd src/
```

No build step required. Check a module for syntax errors with:
```bash
python -c "import ast; ast.parse(open('module.py').read()); print('OK')"
```

---

## Running the Pipeline

### Stage 1 — Extract LLM Embeddings (GPU server)

```bash
# Must be run on GPU server (model weights not stored locally on Mac)
python run_LLM.py
```

Output: `../data/processed/{subDir}/{model_name}_emb_L{layer}_{pooling}_utf8.csv`

### Stage 2 — Nested CV Analysis (Mac)

```bash
python run_nested_cv.py
```

Output:
- `{ParentDir}/{model_name}/ALL_layers/model_corr_layer_roi.json` — correlation per layer × ROI
- `{ParentDir}/{model_name}/L{layer}_{pooling}/K{k}/scatters/` — scatter plots
- Layer-vs-correlation curve plots

---

## Configuration

### `config/data_config.yaml`

```yaml
BrainFile: 'BNL_WPR_brain_behavior.csv'  # fMRI source file (under ../data/raw/)
stimsetid: 'brain-MD1'                    # Stimulus set ID key
Stim2Brain: 'MathMD'                      # Stimulus type: 'MathMD' or 'LangBDS'
Cond_Name: 'WordItem'                     # Condition column name
Cond_values: [0, 1]                       # 0 = math, 1 = text
```

### `config/{model}.json`

```json
{
    "model_name": "FoxBrain_70B_SFT_100K_DPO",
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "AutoModel_config": {
        "model": "../models/...",
        "pooling": "mean",
        "layer": 30
    }
}
```

### Key hyperparameters in `run_nested_cv.py`

```python
seed             = 42
pca_n_components = 80    # PCA dimensionality reduction
k_features       = 20    # SelectKBest top features
roi_List         = [59, 68]  # ROI indices to visualize
```

---

## Module Overview

| Module | Key Class / Function | Role |
|--------|---------------------|------|
| `preprocessing.py` | `ScalePreprocessor` | L2 norm → PCA → SelectKBest; sklearn-compatible |
| `data_module.py` | `FMRIDataModule` | Single shuffle, X/Y alignment by `stimsetid`; no leakage |
| `orchestrator.py` | `AnalysisOrchestrator` | Pure result registry + metrics aggregation (no plotting) |
| `orchestrator.py` | `run_nested_balanced_cv` | Outer + inner CV loop |
| `cv_utils.py` | `generate_balanced_group_splits` | Stratified group splits (balanced C0/C1) |
| `model_adapters.py` | `SklearnAdapter` | Uniform `.fit()` / `.predict()` across frameworks |
| `LLMmodels/embeddings_transf.py` | `load_embeddings_csv` | Load pre-computed embedding CSVs |
| `data/fMRI_data_loader.py` | `Load_dataset` | Load fMRI data from config YAML |
| `utils/metrics.py` | `calculate_voxel_correlation` | Vectorized per-voxel Pearson correlation |
| `utils/plotting.py` | `plot_layer_vs_correlation` | Layer sweep result plots |

---

## Supported LLM Models

Models are selected via `_LLM_CONFIG_MAP` in `run_nested_cv.py`:

| Key | Model | Layers |
|-----|-------|--------|
| `FoxBrain_70B_mean` | FoxBrain-70B-SFT-100K-DPO | 80 |
| `Llama_70B_mean` | Llama-3.3-70B-Instruct | 80 |
| `Qwen3-32B` | Qwen3-32B | 64 |

---

## Architecture Notes

- **No data leakage**: `FMRIDataModule` shuffles once at init; `ScalePreprocessor` fits only on training data.
- **Balanced CV**: `generate_balanced_group_splits` stratifies by `WordItem` (C0/C1) across folds.
- **Separation of concerns**: `AnalysisOrchestrator` holds results only — all plotting lives in `utils/plotting.py`.
- **Circular import guard**: `utils/plotting.py` uses `TYPE_CHECKING` to import `AnalysisOrchestrator` for type hints only.
