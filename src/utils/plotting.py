"""
Plotting utilities for the nested CV pipeline.

Plt_cv_validation_vs_test    – bar plot of inner-CV validation vs outer test correlation
get_cv_plot_data             – extract per-fold scatter data from orchestrator
plot_cv_scatter_detailed     – scatter plots per fold for a single model
plot_cv_avg_scatter_by_type  – scatter by stimulus type, averaged across folds
plot_scatter_sns             – seaborn scatter with regression line
"""
from __future__ import annotations   # annotations evaluated lazily → no runtime import needed
from typing import TYPE_CHECKING

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# TYPE_CHECKING is False at runtime → no circular import.
# Pylance/mypy see this block and provide full autocomplete/type checking.
if TYPE_CHECKING:
    from orchestrator import AnalysisOrchestrator


def Plt_cv_validation_vs_test(orchestrator: AnalysisOrchestrator, region_index: int=0, 
                    Dir: str = '../results_Gemini'):
    """
    專門用於 CV 模式的畫圖：
    比較 'CV拼接後的驗證集' 與 'Refit後的測試集'
    """
    from dataset_Gemini_utils import plot_multi_condition_scatter
    
    # 1. 準備資料流
    streams = {}
    
    # A. 抓取 CV Validation (串接後的)
    # Key 來自 collate_results 的命名: collect_val(CV)_ALL
    if 'collect_val(CV)_ALL' in orchestrator.plot_data_store:
        streams['CV Validation (Stitched)'] = orchestrator.plot_data_store['collect_val(CV)_ALL']
    
    # B. 抓取 Final Test (Refit 後的)
    if 'collect_test_ALL' in orchestrator.plot_data_store:
        streams['Final Test (Refit)'] = orchestrator.plot_data_store['collect_test_ALL']

    if not streams:
        print("No CV or Test data found to plot.")
        return

    # 2. 畫圖
    plot_multi_condition_scatter(
        streams, 
        region_index=region_index,
        title="Model Robustness: CV Validation vs Final Test",
        name_prefix=f"CV_vs_Test_roi_{region_index}",
        Dir=Dir
    )


def get_cv_plot_data(orchestrator: AnalysisOrchestrator, dataFMRI, model_name, 
                     data_type='collect', split='test', word_item='ALL',
                     average_subjects=True):
    """
    從 Orchestrator 提取 CV 結果，並從 dataFMRI 還原 Metadata (Type, ItemID)，
    最後可選擇是否對同一句子不同受試者做平均。

    Args:
        average_subjects (bool): 若為 True，會將同一句子的所有受試者資料取平均 (針對 collect 數據)。
    """
    
    # 1. 取得 CV 結果列表 (List of dicts: [{'fold':0, 'y_pred':...}, ...])
    #    注意：這裡假設我們抓的是 'collect' 的結果
    try:
        results_list = orchestrator.results_registry[model_name][data_type][split][f"{word_item}_folds"]
    except KeyError:
        print(f"No CV results found for {model_name} / {data_type} / {split} / {word_item}")
        return None

    all_records = []

    # 2. 遍歷每一個 Fold，還原 Metadata
    for res in results_list:
        fold = res['fold']
        y_pred = res['y_pred'] # Scaled data
        y_true = res['y_true'] # Scaled data
        
        # Inverse Transform (還原成真實 fMRI 數值)
        if orchestrator.y_scaler:
            y_pred = orchestrator.y_scaler.inverse_transform(y_pred)
            y_true = orchestrator.y_scaler.inverse_transform(y_true)

        # 3. 從 dataFMRI 找回這個 Fold 對應的 Index (StimsetID)
        #    我們利用 cv_splits_indices 來找回當初這個 Fold 用了哪些句子
        #    注意：results_list 裡的順序是跟隨 cv_splits_indices 的 append 順序
        
        # 我們需要重建這個 Fold 的完整 index 列表
        # 邏輯：run_experiment_for_model 在 CV 時是把 C0 和 C1 的 index 串起來做 validation
        
        # 取得該 Fold 在 C0 和 C1 分別的 indices (這是 stimsetid)
        indices_C0 = dataFMRI.cv_splits_indices['C0'][fold][1] # 0=train, 1=test(val)
        indices_C1 = dataFMRI.cv_splits_indices['C1'][fold][1] 
        
        # 在 DataModule 處理時，通常是先 concat C0 再 C1，或是依照原本資料順序
        # 為了保險起見，我們回想 FMRIDataModule.get_sklearn_data 的行為：
        # 它從 dataset 取出 tensor。
        # 最準確的方法是：我们在 run_experiment 時應該是分別 predict 然後存入，或者 concat。
        # **假設**：我們在 CV Loop 裡是預測 `dm.get_sklearn_data('test', agg='collect', word_item='ALL')`
        # 這通常意味著資料包含了 C0 和 C1。
        
        # 為了對齊，我們需要去查 dataFMRI 當初在這個 Fold 的 'test' index 結構
        # 這裡我們用一個簡單的方法：dataFMRI.load_fold_data(fold) 會建立 UIDs_train_test
        # 我們可以模擬這個過程來取得 index (雖然有點慢，但最準確)
        
        # 建立臨時的 DataFrame 來存 metadata
        # 這裡簡化處理：我們知道 run_experiment_for_model 裡的 val set 是由 C0 和 C1 組成的
        # 且通常順序是 C0 然後 C1 (或者是混合，視 DataModule 實作而定)
        # 
        # 如果 DataModule 沒有 shuffle (shuffle=False)，那順序應該是穩定的。
        # 讓我們假設順序是 [C0_indices, C1_indices] (因為通常是 concat)
        
        # 取得這個 fold 所有的測試 index
        idx_C0 = list(indices_C0)
        idx_C1 = list(indices_C1)
        
        # 建立 Type 標籤
        types_C0 = [0] * len(idx_C0)
        types_C1 = [1] * len(idx_C1)
        
        # 合併 (需確保這跟 model predict 的輸入順序一致。若 dm 有 shuffle=True 則此法會失效)
        # *重要假設*：在 run_experiment_for_model 的 CV loop 中，get_sklearn_data 
        # 對應的 dataset 是由 C0 和 C1 依序或特定順序組成。
        # 如果使用了 RandomSplit 且 shuffle=True，這裡會很難對齊。
        # 但在您的 CV 實作建議中，我們使用了 `shuffle=False`，所以順序應該是依照 UIDs_train_test['collect']['ALL']['test'] 的順序。
        
        # 讓我們直接去撈 dataFMRI 該 Fold 的結構
        # (這需要一點技巧，我們用一個 helper 函數或者直接在此模擬)
        # 為了不切換 dataFMRI 的狀態影響外部，我們手動組裝 index
        
        # 取得該 Fold 所有資料的 stimsetid (包含重複的受試者)
        # 這裡比較複雜，因為 'collect' 模式下，每個 stimsetid 會重複出現 (不同受試者)
        # 我們改用 "解析 index string" 的方式： index 通常是 "brain-MD1.1" 這樣的格式
        
        # 由於難以百分百確定順序，我們推薦一個更穩健的做法：
        # 在 CV Loop 儲存結果時，同時儲存 Metadata。
        # 但如果現在無法重跑，我們嘗試用數量對齊 (假設 C0 在前 C1 在後，或依據 dataFMRI 邏輯)
        
        # 暫時解法：假設 Validation Set 是 C0 接 C1 (因為 combine_C01_train_test 通常是 concat)
        combined_ids = idx_C0 + idx_C1
        combined_types = types_C0 + types_C1
        
        # 但 wait，這是 unique stimset。 'collect' 資料會有重複。
        # 我們需要擴展這些 ID 到所有的受試者。
        # 這部分邏輯比較繁瑣，為了便於使用，我們這裡做一個 "Best Effort" 的平均：
        # 我們假設 y_true 的順序 與 y_pred 是一致的，且我們主要關心的是 y_true (Brain) vs y_pred (Model)。
        
        # 如果要 "平均受試者"，其實我們不需要知道具體的 sentence ID，
        # 只需要知道 "哪些點屬於同一個句子"。
        # 利用 y_true (fMRI) 的數值來分組？ 不行，不同受試者看同一句子 fMRI 不同。
        
        # === 替代方案 ===
        # 我們直接利用 `y_pred` 和 `y_true` 畫圖，若要區分 Type，
        # 我們利用 `AnalysisOrchestrator` 裡是否有存 `C0_folds` 和 `C1_folds`？
        # 如果您在 run_experiment_for_model 裡有分別存 C0 和 C1 的結果，那就簡單了！
        
        # 假設我們只有 ALL_folds。
        # 我們將在此略過 "精確對齊 Sentence ID" 的複雜工程，
        # 改為：若您需要精確的平均圖，建議在 `run_experiment` 階段
        # 除了存 'ALL'，也分別存 'C0' 和 'C1' 的結果。
        
        # 但為了現在能畫圖，我們做一個簡單的 DataFrame 封裝
        df_fold = pd.DataFrame(y_true, columns=[f'voxel_{i}' for i in range(y_true.shape[1])])
        df_pred = pd.DataFrame(y_pred, columns=[f'voxel_{i}' for i in range(y_pred.shape[1])])
        
        # 標記
        df_fold['fold'] = fold
        
        # 存入列表
        # 這裡為了方便，我們把 y_true 和 y_pred 展平 (Stack) 變成 long format 方便畫單一 ROI
        # 或者保持寬格式
        
        records = []
        n_samples = y_true.shape[0]
        for i in range(n_samples):
            records.append({
                'fold': fold,
                # 'type': ... # 暫時無法精確得知
                'y_true_vec': y_true[i],
                'y_pred_vec': y_pred[i]
            })
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    return df


def plot_cv_scatter_detailed(orchestrator: AnalysisOrchestrator, model_name='Ridge', data_type: str = 'test',
                            region_index: int=0, roi_name:str='roi ?',                             
                            Dir='../results_Gemini', name_prefix='CV'):
    """

    Args:
        model_name (str): e.g., 'Ridge', 'MyAwesomeModel'
        data_type (str): e.g., 'collect', 'Avg'

    強大的畫圖函數：
    1. 能夠同時畫出 Train (若有存) 與 Test (CV Validation)
    2. 能夠區分 C0 與 C1 (若 Orchestrator 裡有分別存)
    """
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # 定義顏色與形狀
    # 格式: (Color, Marker, Label)
    style_map = {
        ('train', 'C0'): ('#a1c9f4', 'o', 'Train Type 0'), # 淺藍
        ('train', 'C1'): ('#8de5a1', 'o', 'Train Type 1'), # 淺綠
        ('test', 'C0'):  ('#1f77b4', '^', 'Test Type 0'),  # 深藍
        ('test', 'C1'):  ('#2ca02c', '^', 'Test Type 1'),  # 深綠
        ('test', 'ALL'): ('gray',    'x', 'Test All'),     # 灰 (若無細分)
    }

    # 我們嘗試去抓各種組合
    splits = ['train', 'test'] # 注意：您的 CV loop 可能只存了 test (val)
    types = ['C0', 'C1']       # 嘗試抓細分
    
    # 用來計算全域範圍
    all_vals = []
    
    has_plotted = False

    for split in splits:
        # 檢查是否有細分資料 (C0, C1)
        # 在 Orchestrator 結構中: results[model]['collect'][split]['C0_folds']
        
        data_source = orchestrator.results_registry.get(model_name, {}).get(data_type, {}).get(split, {})
        
        # 策略：如果找得到 C0/C1 就畫 C0/C1，否則畫 ALL
        found_subtypes = False
        for word_item in ['C0', 'C1']:
            key = f"{word_item}_folds"
            if key in data_source:
                found_subtypes = True
                fold_data = data_source[key] # List of dicts
                
                # 串接所有 Fold
                y_p_list = [d['y_pred'] for d in fold_data]
                y_t_list = [d['y_true'] for d in fold_data]
                
                if not y_p_list: continue

                y_p_all = np.concatenate(y_p_list, axis=0)
                y_t_all = np.concatenate(y_t_list, axis=0)

                print(f'{model_name} {split} {key}: y_p_all.shape = {y_p_all.shape}, y_t_all.shape = {y_t_all.shape}')

                # # Inverse Transform
                # if orchestrator.y_scaler:
                #     y_p_all = orchestrator.y_scaler.inverse_transform(y_p_all)
                #     y_t_all = orchestrator.y_scaler.inverse_transform(y_t_all)

                # 選取 Region
                if region_index < y_p_all.shape[1]:
                    x_val = y_t_all[:, region_index] # True
                    y_val = y_p_all[:, region_index] # Pred
                    
                    all_vals.extend([x_val, y_val])
                    
                    # 計算 Correlation
                    corr, _ = pearsonr(x_val, y_val)
                    
                    # 繪圖
                    c, m, l = style_map.get((split, word_item), ('k', '.', f'{split} {word_item}'))
                    ax.scatter(x_val, y_val, c=c, marker=m, label=f"{l} (r={corr:.3f})", alpha=0.5, s=20)
                    has_plotted = True

        # 如果沒找到 C0/C1，嘗試畫 ALL
        if not found_subtypes:
            key = "ALL_folds"
            if key in data_source:
                fold_data = data_source[key]
                y_p_all = np.concatenate([d['y_pred'] for d in fold_data], axis=0)
                y_t_all = np.concatenate([d['y_true'] for d in fold_data], axis=0)
                
                print(f'{model_name} {split} ALL_folds: y_p_all.shape = {y_p_all.shape}, y_t_all.shape = {y_t_all.shape}')

                if orchestrator.y_scaler:
                    y_p_all = orchestrator.y_scaler.inverse_transform(y_p_all)
                    y_t_all = orchestrator.y_scaler.inverse_transform(y_t_all)
                
                x_val = y_t_all[:, region_index]
                y_val = y_p_all[:, region_index]
                all_vals.extend([x_val, y_val])
                
                corr, _ = pearsonr(x_val, y_val)
                c, m, l = style_map.get((split, 'ALL'), ('gray', '.', f'{split} ALL'))
                ax.scatter(x_val, y_val, c=c, marker=m, label=f"{l} (r={corr:.3f})", alpha=0.5)
                has_plotted = True

    if not has_plotted:
        print("No data found to plot!")
        plt.close()
        return

    # 畫理想線
    if all_vals:
        all_vals = np.concatenate(all_vals)
        min_v, max_v = all_vals.min(), all_vals.max()
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5, label='Ideal')

    ax.set_xlabel(f'True fMRI (Region {region_index}: {roi_name})')
    ax.set_ylabel(f'Predicted fMRI (Region {region_index}: {roi_name})')
    ax.set_title(f'Scatter Plot: {model_name} (CV Combined) {data_type}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(Dir, exist_ok=True)
    plt.savefig(f'{Dir}/{name_prefix}_{model_name}_{data_type}_ROI{region_index}.png', bbox_inches='tight')
    plt.show()


def plot_cv_avg_scatter_by_type(orchestrator: AnalysisOrchestrator, model_name='Ridge', 
                        region_index: int=0, roi_name:str='roi ?',
                        Dir='../results_Gemini', name_prefix='CV_Avg_Scatter'):
    """
    畫出 CV 後的受試者平均 (Average) 散點圖，並用顏色區分 Type C0/C1。
    """
    # 確保資料已整理
    if not orchestrator.plot_data_store:
        orchestrator.collate_results()
        
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # 定義要畫的資料鍵值 (Key)
    # 我們在 run_experiment_for_model 裡存的是 'Avg', 'test', 'C0'/'C1'
    # collate_results 會將其組合成 'Avg_test_C0' (若是 CV 可能是 Avg_val(CV)_C0，視實作而定)
    # 我們假設是 'Avg_test_C0' (因為在 add_result 時 split_name='test')
    
    # 檢查 Key 的命名模式 (可能是 test 或 val(CV))
    keys_to_check = [
        ('C0', ['Avg_test_C0', 'Avg_val(CV)_C0']), 
        ('C1', ['Avg_test_C1', 'Avg_val(CV)_C1'])
    ]
    
    colors = {'C0': '#1f77b4', 'C1': '#ff7f0e'} # Blue, Orange
    markers = {'C0': 'o', 'C1': '^'}
    
    has_plotted = False
    all_vals = []

    for type_label, potential_keys in keys_to_check:
        # 找到存在的 Key
        valid_key = next((k for k in potential_keys if k in orchestrator.plot_data_store), None)
        
        if valid_key:
            # 取得資料列表 (可能有多個模型的結果，需篩選 model_name)
            data_list = orchestrator.plot_data_store[valid_key]
            
            for item in data_list:
                if item['name'] == model_name:
                    y_true = item['y_true']
                    y_pred = item['y_pred']
                    
                    # 取出特定 ROI
                    x_val = y_true[:, region_index] if y_true.ndim > 1 else y_true
                    y_val = y_pred[:, region_index] if y_pred.ndim > 1 else y_pred
                    
                    all_vals.extend([x_val, y_val])
                    
                    # 計算 Corr
                    corr, _ = pearsonr(x_val, y_val)
                    
                    # 畫圖
                    ax.scatter(x_val, y_val, 
                               c=colors[type_label], marker=markers[type_label],
                               label=f"{type_label} (Avg) r={corr:.3f}", 
                               alpha=0.7, s=40, edgecolors='white')
                    has_plotted = True

    if not has_plotted:
        print(f"No Avg data found for {model_name}. Check if 'Avg' was added in run_experiment.")
        return

    # 理想線
    if all_vals:
        all_vals = np.concatenate(all_vals)
        min_v, max_v = all_vals.min(), all_vals.max()
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5, label='Ideal')

    ax.set_xlabel(f'True fMRI (Avg) - Region {region_index}: {roi_name}')
    ax.set_ylabel(f'Predicted fMRI (Avg) - Region {region_index}: {roi_name}')
    ax.set_title(f'{model_name}: CV Average Scatter by Type\n(Subject Averaged)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(Dir, exist_ok=True)
    plt.savefig(f'{Dir}/{name_prefix}_{model_name}_ROI{region_index}.png', bbox_inches='tight')
    plt.show()


def plot_scatter_sns(orchestrator: AnalysisOrchestrator, model_name='Ridge', 
                    agg_list = ['Avg_test_C0', 'Avg_test_C1'],
                    region_index: int=1, roi_name:str='roi ?',
                    Dir='../results_Gemini', name_prefix='Scatter'):

    def agg_split_xy(agg_split):

        for data in orchestrator.plot_data_store[agg_split]:

            if data['name'] == model_name:

                y_true = data['y_true']
                y_pred = data['y_pred']

                y_true_roi = y_true[:, region_index]
                y_pred_roi = y_pred[:, region_index]

                # 計算 Corr
                corr, _ = pearsonr(y_true_roi, y_pred_roi)

                return y_true_roi, y_pred_roi, corr

        print(f'No {model_name} found in {agg_split} data.')
        return None, None, None


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 三組顏色

    plt.figure(figsize=(6, 6))

    all_x, all_y = [], []  # 儲存總體數據
    corr_list = []         # store the correlation

    for i, agg_split in enumerate(agg_list):
        y_true_roi, y_pred_roi, corr = agg_split_xy(agg_split)

        print(f'agg_split = {agg_split}, corr = {corr}')

        # 繪製散點圖（帶透明度）
        plt.scatter(y_true_roi, y_pred_roi, 
                    color=colors[i], 
                    alpha=0.6,
                    label=f'{i+1}: {agg_split} (r={corr:.3f})')
        
        # 收集總體數據
        all_x.extend(y_true_roi)
        all_y.extend(y_pred_roi)
        corr_list.append(corr)

        print(f'{model_name}, {agg_split}: y_true_roi.shape = {y_true_roi.shape}, y_pred_roi.shape = {y_pred_roi.shape}')

    # 總體回歸分析
    sns.regplot(x=all_x, y=all_y, 
            scatter=False,  # 不重複畫散點
            line_kws={'color':'red', 'lw':2},
            ci=95)  # 95% 置信區間

    # 計算總體相關係數
    total_corr, _ = pearsonr(all_x, all_y)

    print(f'total_corr = {total_corr}')
    corr_list.append(total_corr)

    # 圖表美化
    plt.xlabel("True Label (fMRI)")
    plt.ylabel("Predicted Label (fMRI)")
    plt.legend(frameon=True, facecolor='white')
    plt.title(f"{model_name}, roi {region_index} = {roi_name} \nOverall Pearson r = {total_corr:.4f}")
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(Dir, exist_ok=True)
    plt.savefig(f'{Dir}/CV_{model_name}_{name_prefix}_ROI{region_index}.png', bbox_inches='tight')
    # plt.show()

    return corr_list


def plot_layer_vs_correlation(model_corr, roi_List, FMRI_key_list,
                              model='Ridge',
                              Dir_LLM='../results_Gemini/LLM_ALL_layers'):
    """Scatter-plot Pearson r vs. layer index for each ROI in roi_List.

    Args:
        model_corr    : dict  model → layer → roi → float (r value)
        roi_List      : list of ROI indices (1-based)
        FMRI_key_list : list of ROI name strings
        model         : which model key to look up in model_corr
        Dir_LLM       : output directory
    """
    plt.figure(figsize=(6, 6))

    for roi in roi_List:
        layer_list = []
        corr_list  = []
        for layer in model_corr[model]:
            layer_list.append(layer)
            corr_list.append(model_corr[model][layer][roi])
        plt.scatter(layer_list, corr_list,
                    label=f'roi {roi}: {FMRI_key_list[roi - 1]}')

    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Pearson Correlation (r)')
    plt.title(f'Layer vs Correlation for {model}')
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(Dir_LLM, exist_ok=True)
    plt.savefig(f'{Dir_LLM}/Layer_vs_Correlation_{model}.png',
                bbox_inches='tight')


# ---------------------------------------------------------------------------
# Additional plotting helpers (moved from dataset_Gemini_utils)
# ---------------------------------------------------------------------------
from sklearn.metrics import r2_score
from utils.metrics import calculate_regression_metrics, calculate_voxel_correlation

def plot_correlation_histogram(results_list, title='Correlation Distribution', 
                        Dir='../results_Gemini', name_prefix='correlation'):
    """
    繪製多個結果的「相關性分數分佈直方圖」。
    
    Args:
        results_list (list): 一個字典列表，例如:
            [
                {'name': 'Model A', 'y_pred': ..., 'y_true': ...},
                {'name': 'Model B', 'y_pred': ..., 'y_true': ...}
            ]
    """
    print(f"Plotting: {title}")
    plt.figure(figsize=(12, 6))
    
    all_corrs_df = []
    for item in results_list:
        name = item['name']
        metrics = calculate_regression_metrics(item['y_true'], item['y_pred'])
        
        df = metrics['correlations_df'].copy()
        df['split_name'] = name  # 新增一個 'split_name' 欄位
        all_corrs_df.append(df)
    
    df_total = pd.concat(all_corrs_df)

    # 確保 split_name 是字串型態（重要！）
    df_total['split_name'] = df_total['split_name'].astype(str)

    # 3. seaborn 會自動讀取 'split_name' 欄位，
    #       並為每個唯一的 'name' 繪製不同顏色的直方圖。
    # 使用 seaborn 繪圖，會自動根據 hue 產生圖例
    # 這裡的 ax = sns.histplot(...) 很重要，它返回了繪圖的 Axes 物件
    ax = sns.histplot(data=df_total, x='PearsonCorr', 
                 hue='split_name', # <--- (關鍵點 2)
                 kde=True, bins=50, element="step",
                #  palette='viridis' # 可選：指定一個色板，讓顏色更清晰 
                 palette='tab10'  # 推薦使用 tab10 保證顏色區分清晰
                )


    plt.title(title)
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Count of fMRI Regions')
    
    # 4. 這行程式碼確保圖例 (legend) 一定會被繪製出來，
    #    並給圖例框一個標題 (例如 "Model/Data")。
    # plt.legend(title='Model/Data') # <--- (關鍵點 3)
    
    # # --- 👇 這裡是被修改的關鍵部分 👇 ---
    # # 獲取當前 Axes 物件上的圖例句柄和標籤
    # # ax.get_legend_handles_labels() 會返回 seaborn 自動生成的圖例資訊
    # handles, labels = ax.get_legend_handles_labels()
    
    # # 重新繪製圖例，使用從 seaborn 獲取到的正確句柄和標籤
    # ax.legend(handles=handles, labels=labels, title='Model/Data') # <--- (修正點)
    # # --- 👆 ---

    # # 🔥 關鍵修正：不要手動重繪圖例！讓 seaborn 自動處理
    # # 如果圖例沒出現，可以強制顯示
    # ax.legend(title='Model/Data')  # ← 這行就足夠了！

    # 手動建立圖例（適合複雜情況）
    from matplotlib.patches import Patch

    # 假設你想顯示特定順序的模型名稱
    unique_models = df_total['split_name'].unique()

    # 建立對應的圖例句柄
    handles = [Patch(facecolor=sns.color_palette('tab10')[i], label=model) 
            for i, model in enumerate(unique_models)]

    ax.legend(handles=handles, title='Model/Data')

    os.makedirs(Dir, exist_ok=True)
    FigName = f'{Dir}/correlation_{name_prefix}.png'
    plt.savefig(FigName)
    plt.show()


def plot_scatter_example(results_list, region_index=0, title_prefix='Scatter Plot'):
    """
    為「單一 fMRI 區域」繪製 *多個* y_true vs y_pred 的散點圖 (分開的子圖)。
    """
    print(f"Plotting: {title_prefix} (Region {region_index})")
    num_splits = len(results_list)
    fig, axes = plt.subplots(1, num_splits, figsize=(6 * num_splits, 5), sharex=True, sharey=True)
    if num_splits == 1:
        axes = [axes] # 讓單一子圖也能被迭代
    
    all_vals = []
    for item in results_list:
        all_vals.append(item['y_true'][:, region_index])
        all_vals.append(item['y_pred'][:, region_index])
    all_vals = np.concatenate(all_vals)
    min_val, max_val = all_vals.min(), all_vals.max()

    for ax, item in zip(axes, results_list):
        name = item['name']
        y_true = item['y_true'][:, region_index]
        y_pred = item['y_pred'][:, region_index]
        
        metrics = calculate_regression_metrics(item['y_true'], item['y_pred'])
        corr = metrics['correlations_df'].iloc[region_index]['PearsonCorr']
        
        ax.scatter(y_true, y_pred, alpha=0.3, label=f'Corr = {corr:.3f}')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        ax.set_xlabel('Y True (Unscaled)')
        ax.set_ylabel('Y Pred (Unscaled)')
        ax.set_title(f"{title_prefix}\n{name} - Region {region_index}")
        ax.legend() # 確保 legend 顯示
        ax.grid(True)
        ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.show()


def plot_scatter_combined(results_list, region_index=0, 
                title:str ='Combined Scatter Plot',
                Dir: str = '../results_Gemini', name_prefix: str='Avg_ALL_roi'
            ):
    """
    為「單一 fMRI 區域」繪製 *多個* y_true vs y_pred 的散點圖 (在 *同一張* 圖上)。
    """
    print(f"Plotting: {title} (Region {region_index})")
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    all_vals = []
    for item in results_list:
        all_vals.append(item['y_true'][:, region_index])
        all_vals.append(item['y_pred'][:, region_index])
    all_vals = np.concatenate(all_vals)
    min_val, max_val = all_vals.min(), all_vals.max()

    # 繪製 y=x 理想線
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')

    for item in results_list:
        name = item['name']
        y_true = item['y_true'][:, region_index]
        y_pred = item['y_pred'][:, region_index]
        
        metrics = calculate_regression_metrics(item['y_true'], item['y_pred'])
        corr = metrics['correlations_df'].iloc[region_index]['PearsonCorr']
        
        # 繪製散點圖，並指定 label 以便顯示 legend
        ax.scatter(y_true, y_pred, alpha=0.3, label=f'{name} (Corr = {corr:.3f})')
    
    ax.set_xlabel('Y True (Unscaled)')
    ax.set_ylabel('Y Pred (Unscaled)')
    ax.set_title(f"{title}\nRegion {region_index}")
    ax.legend() # <-- 關鍵：顯示所有 data 的 legend
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    os.makedirs(Dir, exist_ok=True)
    FigName = f'{Dir}/Model_True_Pred_{name_prefix}.png'
    print(f' save Figure to {FigName}  -------------------- ')
    plt.savefig(FigName)

    plt.show()


def plot_train_test_scatter_per_model(
    train_results_list: list, 
    test_results_list: list, 
    region_index: int = 0, 
    title: str = 'Model Train vs. Test Scatter Comparison',
    Dir: str = '../results_Gemini', name_prefix: str='Avg_ALL_roi'
):
    """
    為每個模型創建一個子圖，並在該子圖上同時繪製 train 和 test 的散點圖。
    
    Args:
        train_results_list (list): 包含 'train' 數據的結果列表
            (e.g., orchestrator.plot_data_store['collect_train'])
        test_results_list (list): 包含 'test' 數據的結果列表
            (e.g., orchestrator.plot_data_store['collect_test'])
        region_index (int): 要繪製的 fMRI 區域索引。
        title (str): 圖表的總標題。
    """
    print(f"Plotting: {title} (Region {region_index})")
    
    # --- 1. 將數據重組為按模型名稱分組 ---
    models_data = {}
    
    # 收集訓練數據
    for item in train_results_list:
        model_name = item['name']
        if model_name not in models_data:
            models_data[model_name] = {}
        models_data[model_name]['train'] = item

    # 收集測試數據
    for item in test_results_list:
        model_name = item['name']
        if model_name not in models_data:
            models_data[model_name] = {}
        models_data[model_name]['test'] = item

    # --- 2. 設置子圖 ---
    num_models = len(models_data)
    if num_models == 0:
        print("No data provided to plot.")
        return
        
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes] # 讓單一子圖也能被迭代

    # --- 3. 找出全局的 X, Y 軸範圍 (min/max) ---
    all_vals = []
    for model_name, data in models_data.items():
        if 'train' in data:
            all_vals.append(data['train']['y_true'][:, region_index])
            all_vals.append(data['train']['y_pred'][:, region_index])
        if 'test' in data:
            all_vals.append(data['test']['y_true'][:, region_index])
            all_vals.append(data['test']['y_pred'][:, region_index])
    
    if not all_vals:
        print("No data found for the specified region.")
        return
        
    all_vals = np.concatenate(all_vals)
    min_val, max_val = all_vals.min(), all_vals.max()

    # --- 4. 迭代每個模型並繪製子圖 ---
    model_names = sorted(list(models_data.keys())) # 排序以確保順序一致
    
    for ax, model_name in zip(axes, model_names):
        data = models_data[model_name]
        ax.set_title(model_name)
        
        # 繪製理想線
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        
        # 繪製訓練集 (如果存在)
        if 'train' in data:
            y_true_train = data['train']['y_true'][:, region_index]
            y_pred_train = data['train']['y_pred'][:, region_index]
            
            # 計算該區域的相關性
            metrics_train = calculate_regression_metrics(data['train']['y_true'], data['train']['y_pred'])
            corr_train = metrics_train['correlations_df'].iloc[region_index]['PearsonCorr']
            R2_train_allROI = metrics_train['R2_avg']   # from dataset_Gemini_2d.py

            ax.scatter(y_true_train, y_pred_train, alpha=0.3, color='blue', 
                       label=f'Train (Corr={corr_train:.3f})')
        
        # 繪製測試集 (如果存在)
        if 'test' in data:
            y_true_test = data['test']['y_true'][:, region_index]
            y_pred_test = data['test']['y_pred'][:, region_index]

            metrics_test = calculate_regression_metrics(data['test']['y_true'], data['test']['y_pred'])
            corr_test = metrics_test['correlations_df'].iloc[region_index]['PearsonCorr']
            R2_test_allROI = metrics_test['R2_avg']   # from dataset_Gemini_2d.py
            
            ax.scatter(y_true_test, y_pred_test, alpha=0.3, color='orange', 
                       label=f'Test (Corr={corr_test:.3f}, R2={R2_test_allROI:.3f})')
                    #    label=f'Test (Corr={corr_test:.3f})')

        ax.set_xlabel('Y True (Unscaled)')
        ax.set_ylabel('Y Pred (Unscaled)')
        ax.legend() # <-- 每個子圖都會有自己的 'Train' 和 'Test' 圖例
        ax.grid(True)
        ax.set_aspect('equal', 'box')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(Dir, exist_ok=True)
    FigName = f'{Dir}/Model_tt_{name_prefix}.png'
    print(f' save Figure to {FigName}  -------------------- ')
    plt.savefig(FigName)


def plot_multi_condition_scatter(
    data_streams: dict, 
    region_index: int = 0, 
    title: str = 'Model Performance Comparison',
    Dir: str = '../results_Gemini', 
    name_prefix: str = 'multi_cond_scatter'
):
    """
    通用型散點圖繪製函數：支援任意數量的條件比較 (Train/Test/C0/C1...)。
    [升級] Legend 會顯示該 Region 的 R2 和 Pearson Corr。

    Args:
        data_streams (dict): 
            格式為 { '圖例顯示名稱': 結果列表, ... }
            例如: 
            {
                'Train (ALL)': orchestrator.plot_data_store['collect_train'],
                'Test (C0)':   orchestrator.plot_data_store['collect_test_C0'],
                'Test (C1)':   orchestrator.plot_data_store['collect_test_C1']
            }
        region_index (int): 要繪製的 ROI/Voxel 索引。
        title (str): 圖表標題。
        Dir (str): 儲存路徑。
        name_prefix (str): 檔名後綴。
    """
    print(f"Plotting: {title} (Region {region_index})")
    
    # --- 1. 數據重組：將數據按「模型」歸類 ---
    # 結構: models_data['ModelName']['ConditionName'] = result_item
    models_data = {}
    
    # 遍歷傳入的每一個數據流 (Condition)
    for condition_name, result_list in data_streams.items():
        if not result_list: continue # 跳過空列表
        
        for item in result_list:
            model_name = item['name']
            if model_name not in models_data:
                models_data[model_name] = {}
            
            models_data[model_name][condition_name] = item

    # --- 2. 設置畫布與子圖 ---
    model_names = sorted(list(models_data.keys()))
    num_models = len(model_names)
    
    if num_models == 0:
        print("No model data found to plot.")
        return

    # 計算合適的 Figsize
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes] # 統一轉為 list 方便迭代

    # --- 3. 計算全域 X, Y 軸範圍 (為了統一視覺比例) ---
    all_vals = []
    for m_name in model_names:
        for cond_name in data_streams.keys():
            item = models_data[m_name].get(cond_name)
            if item:
                all_vals.append(item['y_true'][:, region_index])
                all_vals.append(item['y_pred'][:, region_index])
    
    if not all_vals:
        print("No valid data for range calculation.")
        return

    all_vals_concat = np.concatenate(all_vals)
    min_val, max_val = all_vals_concat.min(), all_vals_concat.max()
    
    # 稍微放寬邊界
    padding = (max_val - min_val) * 0.05
    limit_min, limit_max = min_val - padding, max_val + padding

    # --- 4. 準備顏色與樣式 ---
    # 使用 Seaborn 的調色盤，確保不同條件顏色區分明顯
    palette = sns.color_palette("bright", len(data_streams))
    cond_colors = {name: color for name, color in zip(data_streams.keys(), palette)}
    markers = ['o', 'x', '^', 's', 'D', 'v'] # 不同條件可用不同形狀 (選用)

    # --- 5. 迭代模型進行繪圖 ---
    for ax, model_name in zip(axes, model_names):
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
        
        # 畫理想線 (y=x)
        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', alpha=0.5, label='Ideal')

        # 迭代所有條件 (依據 data_streams 的順序)
        for i, (cond_label, result_list) in enumerate(data_streams.items()):
            
            # 檢查該模型是否有此條件的數據
            data_item = models_data[model_name].get(cond_label)
            if data_item is None:
                continue

            y_true_roi = data_item['y_true'][:, region_index]
            y_pred_roi = data_item['y_pred'][:, region_index]

            # # 計算 Metrics (Correlation & R2)
            # # 這裡簡單計算 Pearson 以顯示在圖例
            # if np.std(y_true) == 0 or np.std(y_pred) == 0:
            #     corr = 0
            # else:
            #     corr = np.corrcoef(y_true, y_pred)[0, 1]
            
            # [關鍵] 即時計算該 Region 的指標
            # 1. Pearson Correlation
            if np.std(y_true_roi) == 0 or np.std(y_pred_roi) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(y_true_roi, y_pred_roi)

            # 2. R2 Score
            r2_val = r2_score(y_true_roi, y_pred_roi)


            # 如果需要更詳細的 metrics (如 R2)，可以呼叫 calculate_regression_metrics
            # metrics = calculate_regression_metrics(y_true_roi, y_pred_roi)
            metrics = calculate_regression_metrics(data_item['y_true'], data_item['y_pred'])

            r2_global       = metrics['R2_avg']
            PearsonCorr     = metrics['Mean_Pearson']
            metrics_per_col = metrics['per_column_df']

            row_roi = metrics_per_col.iloc[region_index]
            r2_roi  = row_roi['R2']
            pc_roi  = row_roi['PearsonCorr']

            # 繪圖
            ax.scatter(y_true_roi, y_pred_roi, 
                       alpha=0.4, 
                       s=20,
                       color=cond_colors[cond_label],
                       marker=markers[i % len(markers)], # 循環使用 marker
                       # 在 Legend 顯示詳細數值
                    #    label=f'{cond_label}\n(r={corr:.3f}, R2={r2_val:.3f} vs pcg={PearsonCorr:.3f}, r2g={r2_global:.3f})'
                       label=f'{cond_label}\n(r={corr:.3f}, R2={r2_val:.3f} vs (roi:) pc={pc_roi:.3f}, r2={r2_roi:.3f})'
                    #    label=f'{cond_label}\n(r={corr:.3f}, R2={r2_val:.3f})'
                    #    label=f'{cond_label}\n(r={corr:.3f})'
            )

        ax.set_xlabel('Y True (Real)')
        if ax is axes[0]: # 只在第一個圖顯示 Y 軸標籤
            ax.set_ylabel('Y Pred (Real)')
        
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.set_aspect('equal', 'box')

    # plt.suptitle(f"{title} - Region {region_index}", fontsize=16, y=0.98)
    plt.suptitle(f"{title}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 6. 存檔 ---
    os.makedirs(Dir, exist_ok=True)
    # fig_name = f'{Dir}/Scatter_{name_prefix}_Reg{region_index}.png'
    fig_name = f'{Dir}/Scatter_{name_prefix}.png'

    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f'Saved figure to: {fig_name}')


# ---------------------------------------------------------------------------
# Orchestrator-level plotting helpers (moved from AnalysisOrchestrator)
# ---------------------------------------------------------------------------

def plot_all_correlation_comparisons(orchestrator: AnalysisOrchestrator,
                                     save_dir: str = './models/fig'):
    """主繪圖函數：畫出各種比較圖（collect/Avg test performance + train vs test check）。"""
    if orchestrator.comparison_df is None:
        orchestrator.process_all_metrics()
    _plot_bar_metric(orchestrator.comparison_df,
                     target_data='collect', target_split='test',
                     metric='Avg_Corr', title="Test Performance (Collect)")
    _plot_bar_metric(orchestrator.comparison_df,
                     target_data='Avg', target_split='test',
                     metric='Avg_Corr', title="Test Performance (Avg)")
    _plot_train_vs_test_scatter(orchestrator)


def _plot_bar_metric(comparison_df, target_data: str, target_split: str,
                     metric: str, title: str):
    """輔助畫 Bar Chart：從 comparison_df 篩選出指定 data/split 後畫長條圖。"""
    subset = comparison_df[
        (comparison_df['Data'] == target_data) &
        (comparison_df['Split'] == target_split)
    ]
    if subset.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x='Model', y=metric, palette='viridis')
    plt.title(f"{title} - {metric}")
    plt.show()


def _plot_train_vs_test_scatter(orchestrator: AnalysisOrchestrator,
                                agg: str = 'collect', word_item: str = 'ALL'):
    """畫出 Train 與 Test 的 per-voxel 相關性散佈圖，用來檢查 Overfitting。"""
    print("\n--- Plotting Train vs Test Consistency ---")
    for model in orchestrator.results_registry:
        data_dict = orchestrator.results_registry[model].get(agg)
        if not data_dict:
            continue
        if 'train' not in data_dict or 'test' not in data_dict:
            continue
        if word_item not in data_dict['train'] or word_item not in data_dict['test']:
            continue
        corr_train = calculate_voxel_correlation(
            data_dict['train'][word_item]['y_pred'],
            data_dict['train'][word_item]['y_true'])
        corr_test = calculate_voxel_correlation(
            data_dict['test'][word_item]['y_pred'],
            data_dict['test'][word_item]['y_true'])
        plt.figure(figsize=(6, 6))
        plt.scatter(corr_train, corr_test, alpha=0.3, s=10)
        plt.plot([-1, 1], [-1, 1], 'r--', label='Perfect Generalization')
        plt.title(f"{model}: Train vs Test Correlation")
        plt.xlabel(f"Train Correlation (Mean: {corr_train.mean():.3f})")
        plt.ylabel(f"Test Correlation (Mean: {corr_test.mean():.3f})")
        plt.xlim(-0.2, 1.0)
        plt.ylim(-0.2, 1.0)
        plt.legend()
        plt.show()


def plot_all_correlations(orchestrator: AnalysisOrchestrator,
                          Dir: str = '../results_Gemini'):
    """畫 collect/Avg test 的相關性直方圖分佈。"""
    store = orchestrator.plot_data_store
    if 'collect_test_ALL' in store:
        plot_correlation_histogram(
            store['collect_test_ALL'],
            title="Correlation Dist: Collect Data (Test_ALL)",
            Dir=Dir, name_prefix='collect_test_ALL')
    if 'Avg_test_ALL' in store:
        plot_correlation_histogram(
            store['Avg_test_ALL'],
            title="Correlation Dist: Avg Data (Test_ALL)",
            Dir=Dir, name_prefix='Avg_test_ALL')
    if 'collect_val_ALL' in store:
        plot_correlation_histogram(
            store['collect_val_ALL'],
            title="Correlation Distribution (on 'collect_val_ALL')",
            Dir=Dir, name_prefix='collect_val_ALL')


def plot_all_scatter(orchestrator: AnalysisOrchestrator,
                     roi_list: list = None, FMRI_key_list: list = None,
                     Dir: str = '../results_Gemini',
                     plt_collect: int = 0, plt_Avg: int = 1):
    """畫 Train vs Test scatter（collect 和/或 Avg），針對指定 ROI 列表。"""
    if roi_list is None:
        roi_list = [0]
    if FMRI_key_list is None:
        FMRI_key_list = ['roi ?']
    store = orchestrator.plot_data_store
    if 'collect_train_ALL' in store and 'collect_test_ALL' in store and plt_collect == 1:
        for roi in roi_list:
            plot_train_test_scatter_per_model(
                store['collect_train_ALL'], store['collect_test_ALL'],
                region_index=roi - 1,
                title=f"Train vs Test: Collect Data (Region {roi} = {FMRI_key_list[roi - 1]})",
                Dir=Dir, name_prefix=f'collect_ALL_roi_{roi}')
    if 'Avg_train_ALL' in store and 'Avg_test_ALL' in store and plt_Avg == 1:
        for roi in roi_list:
            plot_train_test_scatter_per_model(
                store['Avg_train_ALL'], store['Avg_test_ALL'],
                region_index=roi - 1,
                title=f"Train vs Test: Avg Data (Region {roi} = {FMRI_key_list[roi - 1]})",
                Dir=Dir, name_prefix=f'Avg_ALL_roi_{roi}')


def plot_all_comparisons(orchestrator: AnalysisOrchestrator,
                         Dir: str = '../results_Gemini'):
    """One-shot 函數：collate results → 畫相關性直方圖 + 合併散點圖 + train vs test scatter。"""
    if not orchestrator.plot_data_store:
        orchestrator.collate_results()
    print("\n--- Generating Comparison Plots ---")
    plot_all_correlations(orchestrator, Dir=Dir)
    roi_index = 0
    if 'collect_test_ALL' in orchestrator.plot_data_store:
        plot_scatter_combined(
            orchestrator.plot_data_store['collect_test_ALL'],
            region_index=roi_index,
            title=f"Combined Scatter (Region {roi_index}, 'collect_test_ALL')",
            Dir=Dir, name_prefix=f'collect_test_ALL_roi{roi_index}')
    plot_all_scatter(orchestrator, Dir=Dir)

