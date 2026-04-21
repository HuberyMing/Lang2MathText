#   Gemini: PyTorch Lightning 數據重構與分析
#
#
#   Q1: 這裏class AnalysisOrchestrator: 的plot_all_comparisons 是繪製比較圖表 (Bar Charts), 
#       除了保留這些function 之外，想像之前一樣有畫更多的狀況, 
#       像是split 有區分'collect' 以及 'Avg' 
#       還有可以畫train 跟test 的結果一起比較之類的, 
#       如何整合這裡新的AnalysisOrchestrator 跟之前舊的AnalysisOrchestrator
#
#   Gemini answer to Q1:
#   這是一個非常好的整合方向！你希望保留 新架構的整潔性 (責任分離)，
#   但同時擁有 舊架構的豐富分析能力 (Train vs Test, Collect vs Avg)。
#
#   關鍵在於：不要讓 AnalysisOrchestrator 負責產生數據（去依賴 X_full 或 deriver_fn），
#   而是讓它負責接收不同來源的「分析結果」。
#
#   我們將舊有的邏輯拆解：
#   1. 數據衍生 (Avg data)：移至 main 流程或 DataModule 處理。
#   2. 多切片推理 (Inference on all splits)：由 ExperimentRunner 或主迴圈執行。
#   3. 彙整與畫圖：保留在 AnalysisOrchestrator，但結構要升級以支援多層次索引 (Model -> DataType -> Split)。
#   
#   以下是整合後的實作：
#   1. 升級版 AnalysisOrchestrator
#       這個版本移除了一切與「原始數據 (X_full)」有關的邏輯，
#       變成了純粹的 結果倉庫與繪圖師。
#
# ------------------------------------------------------------------- #


# Standard library
import copy
import json
import os

# Third-party
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Local – data loading
from LLMmodels.embeddings_transf import load_embeddings_csv

# Local – pipeline modules
from preprocessing import ScalePreprocessor
from data_module import FMRIDataModule

# Local – orchestration
from orchestrator import (
    AnalysisOrchestrator,
    ExperimentRunner,
    collate_cv_results,
    run_nested_balanced_cv,
)

# Local – plotting
from utils.plotting import plot_scatter_sns, plot_layer_vs_correlation

# ---------------------------------------------------------------------------
# Helpers used by the __main__ entry point
# ---------------------------------------------------------------------------

_LLM_CONFIG_MAP = {
    'gpt2-xl':           "../config/gpt2-xl.json",
    'FoxBrain-8B':       "../config/FoxBrain-8B-SFT-166K.json",
    'FoxBrain_70B_last': "../config/FoxBrain_70B_SFT_100K_DPO.json",
    'FoxBrain_70B_mean': "../config/FoxBrain_70B_SFT_100K_DPO.json",
    'Qwen2.5-7B':        "../config/Qwen2.5-7B-Instruct.json",
    'Qwen3-32B':         "../config/Qwen-3-32B.json",
    'Llama_3.2B':        "../config/Llama-3.2-3B-Instruct.json",
    'Llama_70B_old':     "../config/Llama-3.3-70B-Instruct.json",
    'Llama_70B_mean':    "../config/Llama-3.3-70B-Instruct.json",
}


def load_llm_emb(LLM: str, dataFMRI, DirSave: str, layer: int):
    """Load pre-computed embeddings for a single layer from CSV.

    Args:
        LLM       : short model name key (see _LLM_CONFIG_MAP)
        dataFMRI  : loaded fMRI dataset object
        DirSave   : directory containing the embedding CSV files
        layer     : layer index to load

    Returns:
        df_emb      : pd.DataFrame of embeddings (index = stimsetid)
        LLM_info    : info string from metadata
        model_config: parsed JSON config dict
    """
    model_config_path = _LLM_CONFIG_MAP[LLM]
    model_config  = json.load(open(model_config_path, "r"))
    stimset_index = dataFMRI.raw_data["stimset_ALL"].index
    df_emb, meta  = load_embeddings_csv(DirSave, model_config, layer, stimset_index)
    print(f"Loaded layer {layer} embedding: {df_emb.shape}")
    return df_emb, meta["LLM_info"], model_config


def build_output_dir(scaleXY, LLM_setting: str,
                 ParentDir: str = '../results_Gemini') -> str:
    """Build output directory path from LLM_setting and k_features."""
    k_tag = 'noK' if scaleXY.k_features is None else f'K{scaleXY.k_features}'
    return f'{ParentDir}/{LLM_setting}/TestRun/{k_tag}'


if __name__ == "__main__":

    from data.fMRI_data_loader import Load_dataset

    # ------------------------------------------------------------------ #
    #   Configuration
    # ------------------------------------------------------------------ #
    seed             = 42
    pca_n_components = 80
    k_features       = 20

    LLM    = 'Llama_70B_mean'
    subDir = 'Llama-3.3-70B-Instruct_contextMean'

    # ------------------------------------------------------------------ #
    #   Load fMRI dataset
    # ------------------------------------------------------------------ #
    File     = "../config/data_config.yaml"
    dataFMRI, FMRI_key_list = Load_dataset(File, seed=seed, test_size=0.2)

    DirSave = f'../data/processed/{subDir}'

    # ------------------------------------------------------------------ #
    #   Extract raw arrays (used by FMRIDataModule on every layer)
    # ------------------------------------------------------------------ #
    y_raw_df   = dataFMRI.whole['UID_fMRI']
    groups_raw = y_raw_df.index.values
    types_raw  = dataFMRI.whole['UID_WdIt']['WordItem'].values
    assert (groups_raw == dataFMRI.whole['UID_WdIt']['WordItem'].index).all()

    print("--- Raw Data ---")
    print(f"  Y      : {y_raw_df.shape}")
    print(f"  groups : {groups_raw.shape}")
    print(f"  types  : {types_raw.shape}")

    # ------------------------------------------------------------------ #
    #   Sweep over layers
    # ------------------------------------------------------------------ #
    ParentDir  = '../results_Gemini_paper_2g6b'
    roi_List   = [59, 68]
    model_corr = {'Ridge': {}}

    for layer in range(0, 81):
        print(f'\n ------------------- layer = {layer} -----------------')

        df_emb, LLM_info, model_config = load_llm_emb(LLM, dataFMRI, DirSave, layer)

        # Verify coverage
        missing = [g for g in np.unique(groups_raw) if g not in df_emb.index]
        if missing:
            print(f"Warning! Missing embeddings for: {missing}")
        else:
            print("Embedding alignment check passed!")

        source_model = model_config['model_name']
        pooling      = model_config['AutoModel_config']['pooling']
        LLM_setting  = f'{source_model}/L{layer}_{pooling}'

        scaleXY = ScalePreprocessor(
            l2_normalize=True,
            pca_n_components=pca_n_components,
            y_scale=True,
            k_features=k_features,
        )

        dm = FMRIDataModule(
            df_emb=df_emb,
            y_raw=y_raw_df,
            groups=groups_raw,
            types=types_raw,
            seed=seed,
        )
        assert (dm.full_y_raw.index == dm.full_X_raw.index).all(), \
            "DataModule X/Y index mismatch!"

        Dir         = build_output_dir(scaleXY, LLM_setting, ParentDir=ParentDir)
        Dir_scatter = f'{Dir}/scatters'
        Dir_LLM     = f'{ParentDir}/{source_model}/ALL_layers'

        models = [('Ridge', Ridge(alpha=1))]

        orchestrator = run_nested_balanced_cv(models, dm, base_preprocessor=scaleXY)

        for model in ['Ridge']:
            model_corr[model][layer] = {}
            for roi in roi_List:
                corr_list = plot_scatter_sns(
                    orchestrator, model_name=model,
                    agg_list=['Avg_test_C0', 'Avg_test_C1'],
                    region_index=roi - 1,
                    roi_name=FMRI_key_list[roi - 1],
                    Dir=Dir_scatter,
                    name_prefix=f'Scatter_L{layer}',
                )
                print(f'{model} L{layer}, {roi}: {corr_list}')
                model_corr[model][layer][roi] = float(corr_list[-1])

        os.makedirs(Dir_LLM, exist_ok=True)
        with open(f'{Dir_LLM}/model_corr_layer_roi.json', 'w') as f:
            json.dump(model_corr, f, indent=4)

    # ------------------------------------------------------------------ #
    #   Post-sweep: layer-vs-correlation plot + result collation
    # ------------------------------------------------------------------ #
    plot_layer_vs_correlation(model_corr, roi_List, FMRI_key_list,
                              model='Ridge', Dir_LLM=Dir_LLM)

    df = collate_cv_results(orchestrator, model_name='Ridge',
                            data_type='Avg', split_type='test')
    print(df.head())
