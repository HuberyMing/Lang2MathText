"""
Analysis orchestrator and experiment runner for the nested CV pipeline.

AnalysisOrchestrator  – collects per-fold predictions/metrics, aggregates
ExperimentRunner      – sklearn-compatible pipeline runner (fit preprocessor + model per fold)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from preprocessing import ScalePreprocessor
from model_adapters import SklearnAdapter
import copy
from sklearn.base import clone
from cv_utils import generate_balanced_group_splits
from data_module import FMRIDataModule
from utils.metrics import calculate_regression_metrics, calculate_voxel_correlation


class AnalysisOrchestrator:
    def __init__(self):
        # 3層結構: self.results[ModelName][DataType][Split]
        # DataType 例如: 'collect', 'Avg'
        # Split 例如: 'train', 'val', 'test'
        self.results_registry = {} 
        self.y_scaler = None
        self.comparison_df = None
        self.plot_data_store = {} # 快取 collate 後的結果

        # [新] 用來儲存每個模型詳細的 Per-Column (Voxel) 指標，方便後續畫腦圖
        self.detailed_metrics_registry = {}

    def add_result(self, model_name, data_type, split_name, word_item, predictions,
                   targets, scaler=None, fold=None):
        """
        註冊單一結果切片。
        只負責儲存 Raw Result (Scaled)，不做運算，保持輕量。

        Args:
            model_name (str): e.g., 'Ridge', 'MyAwesomeModel'
            data_type (str): e.g., 'collect', 'Avg'
            split_name (str): e.g., 'train', 'test'
            word_item (str): e.g., 'C0', 'C1', 'ALL'
            predictions (np.array): 預測值
            targets (np.array): 真實值
            scaler: (選填) 第一次註冊時傳入
        """
        if model_name not in self.results_registry:
            self.results_registry[model_name] = {}
        
        if data_type not in self.results_registry[model_name]:
            self.results_registry[model_name][data_type] = {}
        
        if split_name not in self.results_registry[model_name][data_type]:
            self.results_registry[model_name][data_type][split_name] = {}

        # metrics = calculate_regression_metrics(targets, predictions)
        # self.results_registry[model_name][data_type][split_name][word_item] = {
        #     'y_pred': predictions,  # 這是 Scaled 的
        #     'y_true': targets       # 這是 Scaled 的
        #     # 'metrics': metrics
        # }

        target_dict = self.results_registry[model_name][data_type][split_name]
        
        # 封裝數據
        data_packet = {'y_pred': predictions, 'y_true': targets}

        if fold is not None:
            # CV 模式：存入 list
            key = f"{word_item}_folds"
            if key not in target_dict: target_dict[key] = []
            data_packet['fold'] = fold
            target_dict[key].append(data_packet)
        else:
            # Standard 模式：直接存
            target_dict[word_item] = data_packet

        # 註冊 Scaler
        if self.y_scaler is None and scaler is not None:
            self.y_scaler = scaler

    def _compute_single_metric(self, y_pred, y_true):
        """Helper: 還原數據並計算指標"""
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_true_inv = self.y_scaler.inverse_transform(y_true)

        # 使用 Unscaled 數據初始化 Analyzer (更通用)
        corrs = calculate_voxel_correlation(y_pred_inv, y_true_inv)

        # 呼叫你的詳細指標計算函數
        metrics = calculate_regression_metrics(y_true_inv, y_pred_inv)
        return {
            'R2': metrics['R2_avg'],
            'Avg_Corr': metrics['Mean_Pearson'],
            'MAE': metrics['MAE'],
            'Mean_Cosine': metrics['Mean_Cosine'],

            'corrs': corrs,

            # 如果你在 Standard 模式下需要詳細的 per_column_df，
            # 可以在這裡回傳，但 process_all_metrics 裡處理 CV 平均時要小心過濾掉它
            'per_column_df': metrics['per_column_df'] 
        }

    def get_region_metrics(self, model_name, condition_key, region_index):
        """
        輔助函數：從儲存庫中獲取特定 Region 的指標 (不用重算)
        condition_key e.g., 'collect_test_ALL'
        """
        if self.comparison_df is None: self.process_all_metrics()
        
        try:
            df = self.detailed_metrics_registry[model_name][condition_key]
            # df 的 index 可能是 ROI 名稱或數字，這裡假設是用 iloc (位置) 存取
            metrics = df.iloc[region_index]
            return metrics # 包含 R2, PearsonCorr, MAE
        except KeyError:
            print(f"Metrics not found for {model_name} - {condition_key}")
            return None

    # ------------------------------------------------------------- #
    #
    #   在這裡實作你的邏輯：CV 模式下移除 DataFrame 以免報錯，
    #                   Standard 模式下保留 DataFrame 以便存檔。
    #
    #   我額外加了一個進階功能：在 CV 模式下，雖然我們從列表移除了 DataFrame，
    #       但我們可以順便計算 「5 個 fold 的平均 Voxel 表現」 並存起來。
    #       這對畫出穩定的 fMRI 腦圖非常有幫助。
    def process_all_metrics(self):

        records = []
        if self.y_scaler is None: raise ValueError("y_scaler not set!")

        # 確保 detailed_metrics_registry 已初始化
        if not hasattr(self, 'detailed_metrics_registry'):
            self.detailed_metrics_registry = {}

        for model_name, types in self.results_registry.items():
            for d_type, splits in types.items():             # d_type = 'collect' or 'Avg' 
                for split_key, words in splits.items():      # split  = 'train' or 'test'

                    # ==========================================
                    # 1. 處理 CV Folds (List)
                    # ==========================================
                    cv_keys = [k for k in words.keys() if k.endswith('_folds')]
                    for key in cv_keys:
                        fold_list = words[key]
                        pure_word_name = key.replace('_folds', '')
                        
                        fold_scalars = []      # 存純量 (R2, MAE...)
                        fold_dfs = []          # 存 DataFrames (per_column_df)
                        
                        for item in fold_list:
                            # 1. 計算所有指標
                            m = self._compute_single_metric(item['y_pred'], item['y_true'])
                            
                            # 2. [關鍵] 使用 pop 移除 DataFrame，避免 DataFrame 建構時崩潰
                            #    同時將其存入 fold_dfs 列表，以備後用
                            df_part = m.pop('per_column_df') 
                            fold_dfs.append(df_part)
                            
                            # 3. 剩下的 m 只包含純量，可以安全加入 list
                            fold_scalars.append(m)
                        
                        # --- 計算純量指標的平均與標準差 ---
                        df_folds_stats = pd.DataFrame(fold_scalars)
                        
                        r2_mean = df_folds_stats['R2'].mean()
                        r2_std = df_folds_stats['R2'].std()
                        corr_mean = df_folds_stats['Avg_Corr'].mean()
                        corr_std = df_folds_stats['Avg_Corr'].std()
                        
                        # --- [進階] 計算 CV 的平均 Voxel 表現 ---
                        # 將 5 個 fold 的 per_column_df 取平均，代表該模型穩定的空間表現
                        # 這非常適合用來畫 "CV Average Brain Map"
                        avg_voxel_df = pd.concat(fold_dfs).groupby(level=0).mean()
                        
                        # 存入 Registry (標記為 CV-Mean)
                        reg_key = f"{d_type}_{split_key}_{pure_word_name} (CV-Mean)"
                        if model_name not in self.detailed_metrics_registry:
                             self.detailed_metrics_registry[model_name] = {}
                        self.detailed_metrics_registry[model_name][reg_key] = avg_voxel_df

                        # 加入紀錄
                        records.append({
                            'Model': model_name,
                            'Data': d_type,
                            'Split': split_key, 
                            'Word_Item': f"{pure_word_name} (CV-Mean)",
                            
                            'R2': r2_mean,
                            'Avg_Corr': corr_mean,
                            'MAE': f"{df_folds_stats['MAE'].mean():.4f}",
                            'R2_Display': f"{r2_mean:.4f} ± {r2_std:.4f}",
                            'Corr_Display': f"{corr_mean:.4f} ± {corr_std:.4f}",
                            'MAE_Display': f"{df_folds_stats['MAE'].mean():.4f} ± {df_folds_stats['MAE'].std():.4f}",
                            'Cosine_Display': f"{df_folds_stats['Mean_Cosine'].mean():.4f} ± { df_folds_stats['Mean_Cosine'].std():.4f}",
                        })

                    # ==========================================
                    # 2. 處理 Standard Split (Dict)
                    # ==========================================
                    std_keys = [k for k in words.keys() if not k.endswith('_folds')]
                    for word_key in std_keys:    # word_key 就是 'ALL', 'C0' 等
                        data = words[word_key]
                        
                        # 1. 計算所有指標
                        m = self._compute_single_metric(data['y_pred'], data['y_true'])
                        
                        # 2. [關鍵] 提取 DataFrame 並存入 Registry
                        per_col_df = m.pop('per_column_df') # 為了保持 records 乾淨，這裡也可以 pop 掉
                        
                        # 存入 Registry
                        reg_key = f"{d_type}_{split_key}_{word_key}"
                        if model_name not in self.detailed_metrics_registry:
                             self.detailed_metrics_registry[model_name] = {}
                        self.detailed_metrics_registry[model_name][reg_key] = per_col_df

                        # 3. 加入紀錄 (m 現在只剩純量了)
                        records.append({
                            'Model': model_name,
                            'Data': d_type,
                            # [關鍵修正點] -----------------------
                            'Split': split_key,    # 這裡應該用外層迴圈的 split_key ('test')
                            'Word_Item': word_key, # 這裡才用內層的 key ('ALL')
                            # -----------------------------------
                            
                            'R2': m['R2'],
                            'Avg_Corr': m['Avg_Corr'],
                            'MAE': f"{m['MAE']:.4f}",
                            # Display 欄位 (僅 Mean)
                            'R2_Display': f"{m['R2']:.4f}",
                            'Corr_Display': f"{m['Avg_Corr']:.4f}",
                            'MAE_Display': f"{m['MAE']:.4f}",
                            'Cosine_Display': m['Mean_Cosine']
                        })

        self.comparison_df = pd.DataFrame(records)
        return self.comparison_df

# ------------------------------------------------------------- #

    def print_final_table(self):
        if self.comparison_df is None:
            self.process_all_metrics()

        df = self.comparison_df

        # Pivot Table: 讓 Model 當 index, (Data+Split) 當 columns
        # 修正 Condition 組合邏輯，包含 Word_Item
        # e.g., "collect_val_ALL (CV-Mean)"
        df['Condition'] = df['Data'] + "_" + df['Split'] + "_" + df['Word_Item']

        # 儲存 df 方便 debug
        self.df = df

        # Pivot
        pivot = df.pivot(
            index='Model', 
            columns='Condition', 
            values=['Avg_Corr', 'R2', 'MAE'] # 確保這裡只選你想看的指標
        )
        self.pivot_table = pivot
        
        print("\n" + "="*100)
        print("FINAL COMPARISON TABLE")
        print("="*100)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', 1000)
        print(pivot)

    def collate_results(self):
        """
        從兩個分析器中收集所有結果，並轉換為繪圖和表格所需的格式。
        將結果整理成適合畫圖 Utils 的格式

        從 Registry 中收集所有結果，並轉換為繪圖所需的格式。
        [修正] 能夠同時處理 Standard (Dict) 和 CV (List) 的數據結構。

        從 Registry 中收集結果。
        關鍵邏輯：
        1. Standard Mode (Dict): 直接取用。
        2. CV Mode (List): 將所有 fold 的結果 Concatenate 起來，形成一個大陣列。
        """
        print("Collating results for visualization...")

        self.plot_data_store = {} 

        for model_name, types in self.results_registry.items():
            for d_type, splits in types.items():        # d_type = 'collect' or 'Avg' 
                for split, words in splits.items():      # split  = 'train' or 'test' or 'val'
                    for raw_word_item, data in words.items():   # raw_word_item = 'ALL', 'C0', 'ALL_folds'
                        
                        # 處理 Key 名稱 (移除 _folds)
                        clean_word_item = raw_word_item.replace('_folds', '')
                        
                        # 這裡我們做一個特殊的標記，如果是 CV 的 val，我們標記為 'val_CV'
                        if raw_word_item.endswith('_folds') and split == 'val':
                            final_split_name = 'val(CV)'
                        else:
                            final_split_name = split

                        split_key = f"{d_type}_{final_split_name}_{clean_word_item}"   # e.g., 'collect_test_C0', 'Avg_test_ALL'

                        if split_key not in self.plot_data_store:
                            self.plot_data_store[split_key] = []

                        # --- 核心邏輯：處理 CV List vs Standard Dict ---
                        try:
                            if isinstance(data, list):
                                # [CV Mode] 串接所有 Folds
                                # data = [{'fold':0, 'y_pred':...}, {'fold':1, 'y_pred':...}]
                                y_pred_scaled = np.concatenate([item['y_pred'] for item in data], axis=0)
                                y_true_scaled = np.concatenate([item['y_true'] for item in data], axis=0)
                            else:
                                # [Standard Mode] 直接取用
                                y_pred_scaled = data['y_pred']
                                y_true_scaled = data['y_true']
                        except Exception as e:
                            print(f"Error collating {model_name} {split_key}: {e}")
                            continue

                        # Inverse Transform
                        if self.y_scaler:
                            y_pred_inv = self.y_scaler.inverse_transform(y_pred_scaled)
                            y_true_inv = self.y_scaler.inverse_transform(y_true_scaled)
                        else:
                            y_pred_inv, y_true_inv = y_pred_scaled, y_true_scaled

                        # 存入 Store
                        self.plot_data_store[split_key].append({
                            'name': model_name,
                            'y_pred': y_pred_inv,
                            'y_true': y_true_inv
                        })
        
        print(f"Collation done. Keys available: {list(self.plot_data_store.keys())}")


class ExperimentRunner:
    def __init__(self, model_name, model_instance, 
                preprocessor: ScalePreprocessor,
                output_dir='./results'):
        self.model_name = model_name
        self.model = model_instance
        self.preprocessor = preprocessor 
        self.adapter = SklearnAdapter(model_instance)

    def run_cv_fold(self, X_train_raw, y_train_raw, X_val_raw, y_val_raw):
        """
        執行流程：
        1. Fit Preprocessor on RAW TRAIN
        2. Transform RAW TRAIN & RAW VAL
        3. Fit Model on PROCESSED TRAIN
        """
        # 1. Fit Preprocessor (Learn PCA, Scaler from Train ONLY)
        # 這是避免 Leakage 的關鍵！
        self.preprocessor.fit(X_train_raw, y_train_raw)
        
        # 2. Transform
        X_train_proc, y_train_proc = self.preprocessor.transform(X_train_raw, y_train_raw)

        # Val 只能 Transform，不能 Fit
        X_val_proc, y_val_proc     = self.preprocessor.transform(X_val_raw, y_val_raw)
        
        # 3. Fit Model
        print(f"[{self.model_name}] Fitting Model (X dim: {X_train_proc.shape[1]})...")
        self.adapter.fit(X_train_proc, y_train_proc)

        return self.adapter, X_val_proc, y_val_proc


def collate_cv_results(orchestrator, model_name='Ridge',
                       data_type: str = 'Avg',
                       split_type: str = 'test') -> pd.DataFrame:
    """Flatten per-fold CV predictions into a tidy DataFrame.

    Args:
        orchestrator : AnalysisOrchestrator with populated results_registry
        model_name   : e.g. 'Ridge'
        data_type    : 'collect' or 'Avg'
        split_type   : 'train' or 'test'

    Returns:
        pd.DataFrame with columns: fold, type, y_true_vec, y_pred_vec
    """
    all_records = []

    folds_data = (orchestrator.results_registry
                  .get(model_name, {})
                  .get(data_type, {})
                  .get(split_type, {}))
    ALL_folds = folds_data.get("ALL_folds", [])
    C0_folds  = folds_data.get("C0_folds", [])
    C1_folds  = folds_data.get("C1_folds", [])

    for ALL_fold, C0_fold, C1_fold in zip(ALL_folds, C0_folds, C1_folds):
        fold       = ALL_fold['fold']
        y_true_ALL = ALL_fold['y_true']
        y_pred_ALL = ALL_fold['y_pred']
        y_true_C0  = C0_fold['y_true']
        y_pred_C0  = C0_fold['y_pred']
        y_true_C1  = C1_fold['y_true']
        y_pred_C1  = C1_fold['y_pred']

        for i in range(y_true_ALL.shape[0]):
            all_records.append({'fold': fold, 'type': 'ALL',
                                'y_true_vec': y_true_ALL[i],
                                'y_pred_vec': y_pred_ALL[i]})
        for i in range(y_true_C0.shape[0]):
            all_records.append({'fold': fold, 'type': 'C0',
                                'y_true_vec': y_true_C0[i],
                                'y_pred_vec': y_pred_C0[i]})
        for i in range(y_true_C1.shape[0]):
            all_records.append({'fold': fold, 'type': 'C1',
                                'y_true_vec': y_true_C1[i],
                                'y_pred_vec': y_pred_C1[i]})

    return pd.DataFrame(all_records)


def run_nested_balanced_cv(models, dm: FMRIDataModule, 
                        base_preprocessor: ScalePreprocessor):
    orchestrator = AnalysisOrchestrator()
    
    # 1. 取得 Raw Data (FMRIDataModule 應只負責 Load Raw Data)
    # 這裡假設 dm.setup() 已經做好了 concat raw data 的工作
    X_all_raw = dm.full_X_raw
    y_all_raw = dm.full_y_raw
    groups_all = dm.full_groups
    types_all = dm.full_types
    
    # 2. Outer Loop Generator
    outer_gen = generate_balanced_group_splits(
        X_all_raw, y_all_raw, groups_all, types_all, 
        n_splits=5, strategy='loocv', seed=42
    )
    outer_splits = list(outer_gen)
    print(f"Starting Perfect Nested CV. Outer Folds: {len(outer_splits)}")
    
    for outer_idx, (out_tr_idx, out_te_idx) in enumerate(outer_splits):
        print(f"\n>>> Outer Fold {outer_idx+1}/{len(outer_splits)} <<<")
        
        # Slice Raw Data
        X_out_tr, y_out_tr = X_all_raw.iloc[out_tr_idx], y_all_raw.iloc[out_tr_idx]
        X_out_te, y_out_te = X_all_raw.iloc[out_te_idx], y_all_raw.iloc[out_te_idx]
        
        # Metadata for Inner Loop & Avg Calculation
        grps_out_tr, types_out_tr = groups_all[out_tr_idx], types_all[out_tr_idx]
        grps_out_te, types_out_te = groups_all[out_te_idx], types_all[out_te_idx]

        for model_name, base_model in models:
            
            # ==========================================
            # Inner Loop (Model Selection / Validation)
            # ==========================================
            inner_gen = generate_balanced_group_splits(
                X_out_tr, y_out_tr, grps_out_tr, types_out_tr,
                n_splits=5, strategy='kfold', seed=42
            )
            
            inner_scores = []
            
            for inner_idx, (in_tr_idx, in_val_idx) in enumerate(inner_gen):
                # Slice Inner Raw Data
                X_in_tr, y_in_tr = X_out_tr.iloc[in_tr_idx], y_out_tr.iloc[in_tr_idx]
                X_in_val, y_in_val = X_out_tr.iloc[in_val_idx], y_out_tr.iloc[in_val_idx]
                
                # Clone Pipeline Components
                curr_preproc = clone(base_preprocessor)
                curr_model = clone(base_model) if hasattr(base_model, 'get_params') else copy.deepcopy(base_model)
                
                # Runner executes: Preproc.fit(Train) -> Preproc.transform -> Model.fit
                runner = ExperimentRunner(f"{model_name}_inner", curr_model, curr_preproc)
                adapter, X_in_val_proc, y_in_val_proc = runner.run_cv_fold(
                    X_in_tr, y_in_tr, X_in_val, y_in_val
                )
                
                # Predict (Result is Scaled)
                y_pred_val_proc = adapter.predict(X_in_val_proc)
                
                # [Metric Logic] 計算 Pearson (需不需要 Inverse Transform 視 Metric 定義而定)
                # Pearson 對線性縮放不敏感，所以用 Scaled data 算也可以，但為了嚴謹我們轉回物理數值
                y_pred_val_phys = curr_preproc.inverse_transform_y(y_pred_val_proc)
                y_true_val_phys = curr_preproc.inverse_transform_y(y_in_val_proc)
                
                # print(f'y_pred_val_phys = {y_pred_val_phys}')
                # print(f'y_true_val_phys = {y_true_val_phys}')
                # print(f'y_pred_val_phys.shape = {y_pred_val_phys.shape}')
                # print(f'y_true_val_phys.shape = {y_true_val_phys.shape}')
                # print(f'y_true_val_phys[:, 2] = {y_true_val_phys.values[:, 2]}')
                # print(f'y_pred_val_phys[:, 2] = {y_pred_val_phys[:, 2]}')
                # print(f'pearsonr = {pearsonr(y_true_val_phys.values[:, 2], y_pred_val_phys[:, 2])}')

                corrs = [pearsonr(y_true_val_phys.values[:, v], y_pred_val_phys[:, v])[0] 
                         for v in range(y_true_val_phys.shape[1])]
                inner_scores.append(np.nanmean(corrs))
            
            print(f"   [{model_name}] Inner CV Avg Pearson: {np.mean(inner_scores):.4f}")

            # ==========================================
            # Refit on Full Outer Train
            # ==========================================
            print("  Refitting on full outer train...")
            
            # 1. 準備全新的 Preprocessor 和 Model
            final_preproc = clone(base_preprocessor)
            final_model = clone(base_model) if hasattr(base_model, 'get_params') else copy.deepcopy(base_model)
            
            runner = ExperimentRunner(model_name, final_model, final_preproc)
            
            # 2. 執行 Refit
            # 這會: final_preproc.fit(X_out_tr) -> transform -> final_model.fit()
            adapter, X_out_te_proc, y_out_te_proc = runner.run_cv_fold(
                X_out_tr, y_out_tr, 
                X_out_te, y_out_te # Test data 這裡只做 transform
            )
            
            # 3. 預測 Outer Test
            y_pred_te_proc = adapter.predict(X_out_te_proc)
            
            # 4. 轉回物理數值 (重要！)
            y_pred_te_phys = final_preproc.inverse_transform_y(y_pred_te_proc)
            y_true_te_phys = final_preproc.inverse_transform_y(y_out_te_proc)
            
            # 5. 存入 'collect' 結果 (Raw Data Level)
            orchestrator.add_result(model_name, 'collect', 'test', 'ALL', 
                                    y_pred_te_phys, y_true_te_phys, 
                                    scaler=None, fold=outer_idx) # Scaler 已處理完，傳 None

            # ==========================================
            # Dynamic Avg Calculation (POC 驗證過的邏輯)
            # ==========================================
            # 使用轉回物理數值的資料來算平均
            df_pred = pd.DataFrame(y_pred_te_phys)
            df_true = pd.DataFrame(y_true_te_phys)
            
            # 加入 Metadata
            df_pred['group'] = grps_out_te
            df_true['group'] = grps_out_te
            
            # GroupBy Mean
            avg_pred_all = df_pred.groupby('group').mean()
            avg_true_all = df_true.groupby('group').mean()
            
            # 復原 Type 資訊
            group_type_map = dict(zip(grps_out_te, types_out_te))
            avg_types = avg_pred_all.index.map(group_type_map)
            
            # 提取數值部分 (去除 index)
            # 假設所有 column 都是數值 (除了 index)
            y_p_avg = avg_pred_all.values
            y_t_avg = avg_true_all.values
            
            # orchestrator.avg_pred_all = avg_pred_all    # for debug
            # orchestrator.avg_true_all = avg_true_all    # for debug
            # orchestrator.group_type_map = group_type_map    # 存下來以便後續分析
            # orchestrator.avg_types = avg_types              # 存下來以便後續分析
            # orchestrator.avg_group_ids = avg_pred_all.index.tolist() # 存下來以便後續分析
            # orchestrator.num_avg_groups = len(avg_pred_all) # 存下來以便後續分析
            # orchestrator.num_avg_regions = y_p_avg.shape[1] # 存下來以便後續分析
            # orchestrator.y_p_avg = y_p_avg              # 存下來以便後續分析
            # orchestrator.y_t_avg = y_t_avg              # 存下來以便後


            # 存入 'Avg' - 'ALL'
            orchestrator.add_result(model_name, 'Avg', 'test', 'ALL', 
                                    y_p_avg, y_t_avg, 
                                    scaler=None, fold=outer_idx)
            
            # 存入 'Avg' - 'C0/C1'
            for c_label in ['C0', 'C1']: # 假設你的 WordItem 是 'C0', 'C1' (In fact 0, 1)
                # mask = (avg_types == c_label)
                mask = (avg_types == int(c_label[1]))  # 'C0'->0, 'C1'->1

                if np.sum(mask) > 0:
                    orchestrator.add_result(model_name, 'Avg', 'test', c_label, 
                                            y_p_avg[mask], y_t_avg[mask], 
                                            scaler=None, fold=outer_idx)

    orchestrator.collate_results()
    return orchestrator

