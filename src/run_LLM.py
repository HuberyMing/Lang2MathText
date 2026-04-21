# %%
#
#   Stage 1 pipeline: Load LLM → extract embeddings → save to data/processed/
#
#   Usage:
#       python run_LLM.py
#
#   This script is intentionally decoupled from any regression or fitting logic.
#   To load saved embeddings for downstream analysis, use:
#       from LLMmodels.embeddings_transf import load_embeddings_csv
#

import os
import json
import time
import logging

from data.fMRI_data_loader import Load_dataset
from LLMmodels.embeddings_transf import run_llm_and_save

logger = logging.getLogger(__name__)


def run_LLM(
    model_config_path,
    stimset,
    DirSave,
    layer_list=None,
    device="Auto",
    manual_batch_size=None,
):
    """
    Load LLM, extract embeddings for all (or specified) layers, save to CSV.

    Each layer is saved as:
        {DirSave}/{model_name}_emb_L{layer}_{pooling}_utf8.csv

    Args:
        model_config_path:  path to model config JSON
        stimset:            DataFrame with 'sentence' column and meaningful index
        DirSave:            directory to save embeddings
        layer_list:         list of layer indices; None = all layers
        device:             'Auto', 'cpu', or 'cuda:N'
        manual_batch_size:  override auto batch-size detection (useful for 70B models)

    Returns:
        num_hidden_layers, hidden_size, gpu_usage
    """
    model_config = json.load(open(model_config_path, "r"))

    gpu_ids = model_config.get("gpu_ids", [])
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"gpu_ids = {gpu_ids}")

    logger.info(f"model_config = {model_config}")

    t0 = time.time()
    _, num_hidden_layers, hidden_size, device, used_devices, gpu_usage = run_llm_and_save(
        model_config,
        DirSave,
        stimset,
        layer_list=layer_list,
        device=device,
        manual_batch_size=manual_batch_size,
    )
    logger.info(f"Total LLM time = {time.time() - t0:.1f}s")
    logger.info(f"device={device}, used_devices={used_devices}")

    return num_hidden_layers, hidden_size, gpu_usage


# %%

if __name__ == "__main__":

    File = "../config/data_config.yaml"
    seed = 42
    dataFMRI, _ = Load_dataset(File, seed=seed, test_size=0.2)
    stimset = dataFMRI.raw_data["stimset_ALL"]

    DirSave = "../data/processed"

    # --- Choose model --- #
    # model_config_path = "../config/gpt2-xl.json";                        subDir = "gpt2-xl_mean"
    # model_config_path = "../config/FoxBrain-8B-SFT-166K.json";           subDir = "FoxBrain-8B_mean"
    model_config_path = "../config/FoxBrain_70B_SFT_100K_DPO.json";     subDir = "FoxBrain_70B_ContextMean_fMRI"
    # model_config_path = "../config/Qwen2.5-7B-Instruct.json";            subDir = "Qwen2.5-7B-Instruct_mean"
    # model_config_path = "../config/Qwen3-32B.json";                      subDir = "Qwen3-32B"
    # model_config_path = "../config/Llama-3.3-70B-Instruct.json";         subDir = "Llama-3.3-70B-Instruct_contextMean"

    DirSave = f"{DirSave}/{subDir}"

    # layer_list=None  →  extract all layers
    # layer_list=[5, 10, 20, 40, 80]  →  extract specific layers only
    run_LLM(
        model_config_path,
        stimset,
        DirSave,
        layer_list=None,
        device="Auto",
    )
