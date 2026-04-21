"""
Cross-validation utility functions.

compute_avg_from_fold           – average per-fold predictions back to stimulus level
get_group_splits                – generate (train_idx, test_idx) pairs by group
generate_balanced_group_splits  – type-balanced grouped splits (inner & outer CV)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.utils import shuffle


def compute_avg_from_fold(y_pred_np, y_true_np, stim_df, group_key='item_id'):
    """
    將 Numpy 格式的預測結果，依照 stim_df 中的 group_key (e.g. 句子ID) 進行平均。
    
    Args:
        y_pred_np (np.array): 模型預測值 (N_samples, N_voxels)
        y_true_np (np.array): 真實值 (N_samples, N_voxels)
        stim_df (pd.DataFrame): 包含 group_key 的 Metadata，長度需與 numpy array 一致
        group_key (str): 用來群組的欄位名 (e.g., 'item_id' 或 'items_num')
    """
    # 1. 建構臨時 DataFrame
    # 使用 stim_df 的 index 確保對齊
    df_pred = pd.DataFrame(y_pred_np, index=stim_df.index)
    df_true = pd.DataFrame(y_true_np, index=stim_df.index)
    
    # 加入 Group ID
    # 注意：需確保 stim_df 的順序與 y_pred_np 完全對應 (get_sklearn_data 保證了這一點)
    group_ids = stim_df[group_key].values
    
    df_pred['__gid__'] = group_ids
    df_true['__gid__'] = group_ids
    
    # 2. GroupBy Mean
    # 預設 sort=True，這會讓結果依照 item_id 排序
    avg_pred = df_pred.groupby('__gid__').mean()
    avg_true = df_true.groupby('__gid__').mean()
    
    return avg_pred.values, avg_true.values


def get_group_splits(groups, n_splits=5, method='kfold', seed=42):
    """
    根據 Group (e.g., stimsetid) 產生索引生成器。
    
    Args:
        groups (np.array): 形狀為 (n_samples,) 的陣列，記錄每筆數據的 Group ID
        n_splits (int): Fold 數
        method (str): 'kfold' 或 'loocv'
    
    Yields:
        (train_idx, test_idx): 每一折的索引
    """
    if method == 'loocv':
        # Leave-One-Group-Out (外層常用)
        # 注意：這會產生與 Group 數量一樣多的 Fold
        cv = LeaveOneGroupOut()
        return cv.split(np.zeros(len(groups)), groups=groups)
    
    else:
        # Group K-Fold (內層常用)
        # 注意：GroupKFold 不支援 shuffle 參數，若需隨機性，需在外部先 shuffle groups (這裡簡化處理)
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(np.zeros(len(groups)), groups=groups)


def generate_balanced_group_splits(X, y, groups, types, n_splits=5, strategy='kfold', seed=42):
    """
    產生平衡的群組切分索引。
    
    邏輯：
    1. 對每個 Type (e.g., C0, C1) 分別找出獨特的 Groups (Sentences)。
    2. 對每個 Type 的 Groups 進行切分 (Train/Test Group lists)。
    3. 將各 Type 的第 k 折 Train Groups 合併，Test Groups 合併。
    4. 將 Groups 映射回原始數據的 Indices。

    Args:
        groups (np.array): 每個樣本的 Group ID (句子ID)。
        types (np.array): 每個樣本的 Type ID (WordItem)。
        n_splits (int): K-Fold 的折數 (LOOCV 時此參數無效)。
        strategy (str): 'kfold' 或 'loocv'。
        seed (int): 隨機種子 (用於 Shuffle Groups)。

    Yields:
        (train_idx, test_idx): 原始數據層級的索引。
    """
    
    # 1. 整理每個 Type 下的 Unique Groups
    # 結構: type_group_map = { 'C0': [g1, g2, ...], 'C1': [g10, g11, ...] }
    unique_types = np.unique(types)
    type_group_map = {}
    
    rng = np.random.RandomState(seed)
    
    for t in unique_types:
        # 找出屬於該 Type 的所有樣本的 groups
        t_mask = (types == t)
        t_groups = np.unique(groups[t_mask])
        
        # Shuffle 確保隨機性 (對於 K-Fold 很重要，對於 LOOCV 若數量不對等也很重要)
        rng.shuffle(t_groups)
        type_group_map[t] = t_groups

    # 2. 決定折數 (Num Folds)
    if strategy == 'loocv':
        # LOOCV: 折數取決於「最少句子的那個 Type 有幾句」
        # 例如 C0 有 50 句, C1 有 60 句 -> 只能跑 50 折 (為了嚴格配對)
        # 多的 10 句 C1 會在每次配對中被隨機分配到 Train 或被捨棄 (視實作而定)
        # 這裡採取「截斷對齊」策略：只跑 min_len 折
        min_len = min([len(g) for g in type_group_map.values()])
        actual_n_splits = min_len
        splitter_cls = KFold # 用 KFold(n=樣本數) 模擬 LOO，方便控制數量
    else:
        actual_n_splits = n_splits
        splitter_cls = KFold

    print(f"   [BalancedSplit] Strategy={strategy}, Total Folds={actual_n_splits}")

    # 3. 針對每個 Type 產生 Group 的切分 (List of splits)
    # type_splits[t] = [ (train_groups_fold1, test_groups_fold1), (train_fold2, test_fold2)... ]
    type_splits = {}
    
    for t, t_groups in type_group_map.items():
        # 如果是 LOOCV，我們用 KFold(n_splits=總數) 來模擬 Leave-One-Out
        # 如果是 K-Fold，直接用 KFold
        
        # 注意：若 groups 數量小於 n_splits，KFold 會報錯，需防呆
        current_n = min(actual_n_splits, len(t_groups))
        
        kf = splitter_cls(n_splits=current_n)
        
        # 儲存該 Type 每一折選到的 "Group ID"
        splits_for_this_type = []
        for train_idx_g, test_idx_g in kf.split(t_groups):
            train_groups = t_groups[train_idx_g]
            test_groups = t_groups[test_idx_g]
            splits_for_this_type.append((train_groups, test_groups))
            
        type_splits[t] = splits_for_this_type

    # 4. 縫合 (Zip) 各 Type 的切分並轉回原始 Index
    for i in range(actual_n_splits):
        
        combined_train_groups = []
        combined_test_groups = []
        
        for t in unique_types:
            # 取出 Type t 在第 i 折的 group 分配
            # 若某 Type 數量不足 (例如 fold 數 > 該 Type 句子數)，這里取模數循環使用
            fold_data = type_splits[t][i % len(type_splits[t])]
            
            combined_train_groups.extend(fold_data[0])
            combined_test_groups.extend(fold_data[1])
            
        # 5. 將 Group ID 轉回 Data Sample Index
        # 使用 np.isin 快速查找
        train_mask = np.isin(groups, combined_train_groups)
        test_mask  = np.isin(groups, combined_test_groups)
        
        # 轉成整數索引
        yield np.where(train_mask)[0], np.where(test_mask)[0]



def combine_data_streams(stream_a, stream_b, new_suffix="Combined"):
    """
    將兩個結果列表合併 (Concatenate)，通常用於合併 Train 和 Test。
    
    Args:
        stream_a (list): 結果列表 A (e.g., Train data)
        stream_b (list): 結果列表 B (e.g., Test data)
        new_suffix (str): (選用) 用於標記，目前主要用於 debug。
        
    Returns:
        list: 合併後的結果列表，格式與輸入相同。
    """
    merged_list = []
    
    # 建立查找表，以模型名稱為 Key，方便快速對應
    # 假設 stream_b 的模型名稱是唯一的
    dict_b = {item['name']: item for item in stream_b}
    
    for item_a in stream_a:
        name = item_a['name']
        
        # 只有當 B 列表也有這個模型時才合併
        if name in dict_b:
            item_b = dict_b[name]
            
            # --- 核心邏輯：串接 Numpy Array ---
            # axis=0 代表在筆數 (Sample) 方向串接
            merged_y_pred = np.concatenate([item_a['y_pred'], item_b['y_pred']], axis=0)
            merged_y_true = np.concatenate([item_a['y_true'], item_b['y_true']], axis=0)
            
            # 建立新的結果物件
            new_item = {
                'name': name,
                'y_pred': merged_y_pred,
                'y_true': merged_y_true,
                # 如果有其他 metadata，通常取其中一份的即可，或者忽略
            }
            merged_list.append(new_item)
        else:
            print(f"Warning: Model '{name}' found in Stream A but not in Stream B. Skipping merge.")
            
    return merged_list

