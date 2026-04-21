
import pandas as pd
import numpy as np
import torch

# 導入我們需要的所有評估指標函式
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Metrics:
    @classmethod
    def r2_score(cls, y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    @classmethod
    def mse(cls, y_true, y_pred):
    
        mse_self = cls.mse_self(y_true, y_pred)
        # print(mse_self)
        # print(mean_squared_error(y_true, y_pred))
        assert np.allclose(mse_self, mean_squared_error(y_true, y_pred))

        return mean_squared_error(y_true, y_pred)
    
    @classmethod
    def mae(cls, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)


    @classmethod
    def squared_pearson_corr(cls, y_true: np.ndarray, y_pred: np.ndarray):
        return np.corrcoef(y_true, y_pred)[0, 1] ** 2
    
    @classmethod
    def rmse(cls, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @classmethod
    def mae_self(cls, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.abs(y_true - y_pred))
    
    # @classmethod
    # def mse_self(cls, y_true: np.ndarray, y_pred: np.ndarray):
    #     return np.mean((y_true - y_pred) ** 2)

    @classmethod
    def mse(cls, y_true, y_pred):
        # 確保輸入是PyTorch張量
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)

        # 計算均方誤差
        mse_value = torch.mean((y_true - y_pred) ** 2)
        
        return mse_value.item()  # 返回均方誤差

#
#   get fitting score
#
def df_corre_2_str(map_cls, df_corre_Plt, roi_plot):
            
    if map_cls == 'ridgeCV':
        metric = ['Pearson', 'p (Pears)', 'full alpha']

        roi_corre = df_corre_Plt.loc[roi_plot].iloc[[0,1,4]]

        assert (roi_corre.index == metric).all()
        Pearson, p, alpha = roi_corre.values

        Metric_corr = f'(Pearson, p, alpha)'
        Value_corr  = f'({Pearson:.2f}, {p:.2f}, {alpha})'

    else:
        metric = ['Pearson', 'p (Pears)']

        roi_corre = df_corre_Plt.loc[roi_plot].iloc[[0,1]]

        assert (roi_corre.index == metric).all()
        Pearson, p = roi_corre.values

        Metric_corr = f'(Pearson, p)'
        Value_corr  = f'({Pearson:.2f}, {p:.2f})'

    return Metric_corr, Value_corr

def roi_data(Y_predict_Set1, Y_data_Set1, df_corre_Set1, score_Set1, 
             Y_predict_Set2, Y_data_Set2, df_corre_Set2, score_Set2, 
             roi_plot, map_cls):

    #  Set 1, usually = Train (i.e. Fit)  or  (cond = 0)
    pred_Set1 = Y_predict_Set1[roi_plot]
    data_Set1 = Y_data_Set1[roi_plot]

    #  Set 2, usually = Test              or  (cond = 1)
    pred_Set2  = Y_predict_Set2[roi_plot]
    data_Set2  = Y_data_Set2[roi_plot]

    Metric_Set1, Value_Set1 = df_corre_2_str(map_cls, df_corre_Set1, roi_plot)
    Metric_Set2, Value_Set2 = df_corre_2_str(map_cls, df_corre_Set2, roi_plot)

    data_pred = ('prediction', pred_Set1, pred_Set2)
    data_fMRI = ('fMRI data',  data_Set1, data_Set2)
    scores    = (score_Set1, score_Set2)

    assert Metric_Set1 == Metric_Set2
    str_corre = (Metric_Set1, Value_Set1, Value_Set2)

    return data_pred, data_fMRI, scores, str_corre


#   ****  mapping (X <--> Y)  *****  #
#   ****  mapping (X <--> Y)  *****  #
#   ****  mapping (X <--> Y)  *****  #

def predict_2_correlation(model, X_df_emb, Y_data, y_scaler=None, **kwargs):
    # ------------------------------------- #
    #       do the predition                #
    # ------------------------------------- #
    assert (X_df_emb.index == Y_data.index).all()
    # print(f'y_scaler = {y_scaler}')
    # print(f'y_scaler.inverse_transform = {y_scaler.inverse_transform}')

    if y_scaler == None:
        Y_predictions  = mapping_predict(model, X_df_emb, Y_data.columns)
    else:
        # Y_pred_scaled  = mapping_predict(model, X_df_emb, Y_data.columns)
        Y_pred_scaled  = model.predict(X_df_emb)

        Y_predictions  = y_scaler.inverse_transform(Y_pred_scaled)

        Y_predictions = pd.DataFrame(Y_predictions, index=X_df_emb.index, columns=Y_data.columns)


    # --------- find correlation between predictions and neural data --------- #
    df_correlations, corr_raw, p_raw = get_correlation_Pearson(Y_predictions, Y_data, model)

    df_corre_Pears = statistics_correlation_Nan(Y_predictions, Y_data, corr_raw, p_raw, model)


    score_wrong = model.score(X_df_emb, Y_data)

    r2_score = Metrics.r2_score(Y_data, Y_predictions)

    print(f'        r2_score = {r2_score} vs  score_wrong(at scaled space) = {score_wrong}  -----------')
    # assert score_wrong == r2_score

    return df_corre_Pears, Y_predictions, r2_score


# ------------------------------------- #
#       statistical correlation         #
# ------------------------------------- #

def statistics_correlation_Nan(predictions, neural_data, corr_raw, p_raw, model):

    correlations = [corr if not np.isnan(corr) else 0 for corr in corr_raw]
    p_scores = [p if not np.isnan(p) else 1.0 for p in p_raw]

    std_pred = [np.std(predictions.iloc[:, i]) for i in range(predictions.shape[1])]
    std_data = [np.std(neural_data.iloc[:, i]) for i in range(neural_data.shape[1])]

    Tol = 1e-10 # Tolerance for numerical stability
    correlations = [corr if std_pred[i] > Tol and std_data[i] > Tol else 0 for i, corr in enumerate(correlations)]
    p_scores = [p if std_pred[i] > Tol and std_data[i] > Tol else 1.0 for i, p in enumerate(p_scores)]

    df_correlations = pd.DataFrame({
        'Pearson': correlations,
        'p (Pears)': p_scores,
        'Pears raw': corr_raw,
        'p raw': p_raw,
        'full alpha': model.alpha_ if hasattr(model, 'alpha_') else np.nan,
    }, index=neural_data.columns)

    return df_correlations



def get_correlation_Pearson(predictions, neural_data, model):
    # --------- find correlation betweend predictions and neural data --------- #
    from scipy.stats import pearsonr

    print(f'        predictions.shape = {predictions.shape}')
    print(f'        neural_data.shape = {neural_data.shape}')

    # 計算相關係數
    corr_Pears = []
    p_Pears = []
    std_err  = 1e-10  

    corr_raw = []
    p_raw = []

    for i in range(predictions.shape[1]):
        pred = predictions.iloc[:, i]
        orig = neural_data.iloc[:, i]

        corr, p = pearsonr(pred, orig)

        corr_raw.append(corr)
        p_raw.append(p)

        # 檢查標準差是否過低
        if np.std(pred) < std_err or np.std(orig) < std_err:
            print(f"Warning: Standard deviation too low for column {i}, setting correlation to 0")
            corr = 0.0
            p = 1.0

        # 檢查是否為 NaN
        if np.isnan(corr):
            print(f"Warning: NaN correlation for column {i}")
            corr = 0.0
        if np.isnan(p):
            print(f"Warning: NaN p-score for column {i}")
            p = 1.0

        corr_Pears.append(corr)
        p_Pears.append(p)
    # print(f'Pearson: {corr_Pears}')  
    # print(f'p (Pears): {p_Pears}')

    # 將相關係數轉換為DataFrame
    df_correlations = pd.DataFrame(corr_Pears, columns=['Pearson'])
    df_correlations.index = neural_data.columns
    df_correlations['p (Pears)'] = p_Pears
    df_correlations['full alpha'] = model.alpha_ if hasattr(model, 'alpha_') else None
    # df_correlations = df_correlations.sort_values(by='Correlation', ascending=False)

    return df_correlations, corr_raw, p_raw


#
#   model prediction
#

def mapping_predict(model, df_emb, columns):
    """Predict or transform the neural data using the model and embeddings.
    Args:
        model: The mapping model (e.g., Ridge, LinearRegression).
        df_emb: DataFrame containing the embeddings.
        neural_data: DataFrame containing the neural data.
    Returns:
        predictions: DataFrame containing the predicted neural data.
    """

    # Predict or transform as needed
    predictions = model.predict(df_emb)
    print(f'        predictions.shape = {predictions.shape}')

    predictions = pd.DataFrame(predictions, index=df_emb.index, columns=columns)

    return predictions


# ---------------------------------------------------------------------------
# Regression metric helpers (moved from dataset_Gemini_utils)
# ---------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import torch

def calculate_regression_metrics_ALLavg(y_true, y_pred):
    """
    一個可重用的函數，用於計算所有標準迴歸指標。
    y_true 和 y_pred 必須是 unscaled 的 numpy arrays。
    """
    
    # --- 1. 計算「平均」指標 (在所有 voxels/outputs 上取平均) ---
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² (R-squared) - 'uniform_average' 是多輸出迴歸的標準
    r2_avg = r2_score(y_true, y_pred, multioutput='uniform_average')

    # --- 2. 計算「逐-Voxel」指標，然後取平均 ---
    correlations = []
    cosine_sims = []
    
    num_voxels = y_true.shape[1]
    for i in range(num_voxels):
        y_t_voxel = y_true[:, i]
        y_p_voxel = y_pred[:, i]
        
        # Pearson Correlation
        # 處理 y_pred 或 y_true 是常數的罕見情況
        if np.std(y_t_voxel) == 0 or np.std(y_p_voxel) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(y_t_voxel, y_p_voxel)
        correlations.append(corr)
        
        # Cosine Similarity
        sim = cosine_similarity(y_t_voxel.reshape(1, -1), y_p_voxel.reshape(1, -1))[0, 0]
        cosine_sims.append(sim)

    # 處理 NaN
    correlations = np.nan_to_num(correlations) 
    cosine_sims = np.nan_to_num(cosine_sims)

    mean_pearson = np.mean(correlations)
    median_pearson = np.median(correlations)
    mean_cosine = np.mean(cosine_sims)
    
    return {
        'MAE': mae,
        'R2_avg': r2_avg,
        'Mean_Pearson': mean_pearson,
        'Median_Pearson': median_pearson,
        'Mean_Cosine': mean_cosine,
        # 保留原始的 DataFrame 以便未來繪圖
        'correlations_df': pd.DataFrame(correlations, columns=['PearsonCorr'])
    }


def calculate_regression_metrics(y_true, y_pred, column_names=None):
    """
    計算詳細的迴歸指標。
    y_true, y_pred: (n_samples, n_targets) 的 numpy arrays (unscaled)。
    column_names: (可選) 每個 target 的名稱列表，用於 DataFrame index。
    """

    # 安全轉換：支援 tensor / ndarray / list 等
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # # 轉為 tensor（若還在 GPU 上可保留加速）
    # if isinstance(y_true, np.ndarray):
    #     y_true = torch.from_numpy(y_true).float()
    # else:
    #     y_true = y_true.float()

    # if isinstance(y_pred, np.ndarray):
    #     y_pred = torch.from_numpy(y_pred).float()
    # else:
    #     y_pred = y_pred.float()

    # # 確保 y_true 和 y_pred 是 PyTorch 張量
    # if not isinstance(y_true, torch.Tensor):
    #     y_true = torch.tensor(y_true)
    # if not isinstance(y_pred, torch.Tensor):
    #     y_pred = torch.tensor(y_pred)

    # --- 0. 計算「平均」指標 (在所有 voxels/outputs 上取平均) ---
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # --- 1. R-squared (R2) ---
    # 'raw_values': 返回 shape 為 (n_targets,) 的 array，包含每個 column 的 R2
    r2_raw = r2_score(y_true, y_pred, multioutput='raw_values')
    
    # 'uniform_average': 等同於 np.mean(r2_raw)
    r2_avg_global = np.mean(r2_raw) 

    # --- 2. Pearson & Cosine (Per Column) ---
    correlations = []
    cosine_sims = []
    mae_raw = [] # 我們也算每個 column 的 MAE
    
    num_targets = y_true.shape[1]
    
    for i in range(num_targets):
        y_t_col = y_true[:, i]
        y_p_col = y_pred[:, i]
        
        # Pearson
        if np.std(y_t_col) == 0 or np.std(y_p_col) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(y_t_col, y_p_col)
        correlations.append(corr)
        
        # Cosine Similarity
        # reshape(1, -1) 是因為 sklearn 期望 (n_samples, n_features)
        # 這裡我們把整個 column 當作一個 sample 向量來比較相似度
        sim = cosine_similarity(y_t_col.reshape(1, -1), y_p_col.reshape(1, -1))[0, 0]
        cosine_sims.append(sim)
        
        # MAE per column
        mae_raw.append(mean_absolute_error(y_t_col, y_p_col))

    # 處理潛在的 NaN
    correlations = np.nan_to_num(np.array(correlations))
    cosine_sims = np.nan_to_num(np.array(cosine_sims))
    mae_raw = np.array(mae_raw)

    # --- 3. 建立 Per-Column DataFrame ---
    metrics_per_col = pd.DataFrame({
        'R2': r2_raw,
        'PearsonCorr': correlations,
        'CosineSim': cosine_sims,
        'MAE': mae_raw
    })
    
    if column_names is not None and len(column_names) == num_targets:
        metrics_per_col.index = column_names

    # --- 4. 彙總 Global Metrics ---
    global_metrics = {
        'MAE': mae,
        'MAE_avg': np.mean(mae_raw),
        'R2_avg': r2_avg_global,
        'Mean_Pearson': np.mean(correlations),
        'Median_Pearson': np.median(correlations),
        'Mean_Cosine': np.mean(cosine_sims),
        # 保留原始的 DataFrame 以便未來繪圖
        'correlations_df': pd.DataFrame(correlations, columns=['PearsonCorr'])
    }
    
    return {
        # 為了相容性，我們把主要指標直接放在外層 (或 metrics key 下)
        **global_metrics, 
        'per_column_df': metrics_per_col # 這包含了所有細節
    }


def calculate_voxel_correlation(y_pred, y_true) -> np.ndarray:
    """Vectorised per-voxel Pearson correlation between predictions and targets.

    Args:
        y_pred : array-like (n_samples, n_voxels) — model predictions
        y_true : array-like (n_samples, n_voxels) — ground-truth fMRI values

    Returns:
        corr : np.ndarray of shape (n_voxels,) — Pearson r per voxel
    """
    import torch as _torch
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, _torch.Tensor) else np.asarray(y_pred)
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, _torch.Tensor) else np.asarray(y_true)

    pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    true_c = y_true - y_true.mean(axis=0, keepdims=True)

    numerator   = np.sum(pred_c * true_c, axis=0)
    denom_pred  = np.sqrt(np.sum(pred_c ** 2, axis=0))
    denom_true  = np.sqrt(np.sum(true_c ** 2, axis=0))

    return np.divide(numerator, denom_pred * denom_true,
                     out=np.zeros_like(numerator),
                     where=(denom_pred * denom_true) != 0)
