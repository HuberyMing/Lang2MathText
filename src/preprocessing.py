"""
Feature preprocessing pipeline for LLM embeddings + fMRI targets.

Exports
-------
f_regression_multi_output : score function for SelectKBest with multi-output y
ScalePreprocessor         : sklearn-compatible transformer (L2 norm → PCA → SelectKBest)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler


def f_regression_multi_output(X, y):
    f_scores = [f_regression(X, y[:, i])[0] for i in range(y.shape[1])]
    return np.mean(f_scores, axis=0), np.full(X.shape[1], 0.01)


class ScalePreprocessor(BaseEstimator, TransformerMixin):
    """
    負責管理 X (Scaling, PCA, Feature Selection) 和 Y (Scaling) 的 Pipeline。
    完全遵循 fit/transform 模式，避免 Data Leakage。

    Parameters
    ----------
    l2_normalize : bool
        Row-wise L2 normalise embeddings before PCA. Default True.
    pca_n_components : int or float or None
        Passed directly to sklearn PCA. Float = variance ratio, int = n dims,
        None / 0 = skip PCA. Default 0.95.
    y_scale : bool
        Standardise (zero-mean, unit-variance) fMRI targets. Default True.
    k_features : int or None
        Keep top-k features after PCA via SelectKBest. None = skip. Default None.
    """
    def __init__(self,
                 l2_normalize: bool = True,
                 pca_n_components=0.95,
                 y_scale: bool = True,
                 k_features: int = None,
                 ):

        self.l2_normalize = l2_normalize
        self.pca_n_components = pca_n_components
        self.y_scale = y_scale
        self.k_features = k_features

        # 內部模型狀態 (初始化為 None)
        self.pca = None
        self.selector_reg = None
        self.y_scaler_ = None
        self.is_fitted = False

        # 記錄資訊用
        self.info_str = ""

    def fit(self, X, y):
        """
        在 Training Data 上學習參數 (PCA components, Scaler mean/std, Best Features)。
        X: Raw Embeddings
        y: Raw fMRI signals
        """
        # 1. X L2 Norm (Row-wise, stateless)
        X_curr = self._apply_l2(X)

        # 2. Y Scaling (Fit)
        y_scaled = y
        if self.y_scale:
            self.y_scaler_ = StandardScaler()
            y_scaled = self.y_scaler_.fit_transform(y)

        # 3. PCA (Fit & Transform for next step)
        if self.pca_n_components and self.pca_n_components != 0:
            self.pca = PCA(n_components=self.pca_n_components)
            X_curr = self.pca.fit_transform(X_curr)

            dim_info = self.pca.n_components_
            if isinstance(self.pca_n_components, float):
                var_sum = np.sum(self.pca.explained_variance_ratio_)
                self.info_str += f"[PCA: var={self.pca_n_components}({var_sum:.2%}) -> dim={dim_info}]"
            else:
                self.info_str += f"[PCA: n={self.pca_n_components}]"

        # 4. Feature Selection (Fit)
        if self.k_features:
            self.selector_reg = SelectKBest(score_func=f_regression_multi_output, k=self.k_features)
            self.selector_reg.fit(X_curr, y_scaled)
            self.info_str += f" -> [SelectK: k={self.k_features}]"

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        if not self.is_fitted:
            raise RuntimeError("ScalePreprocessor is not fitted. Call fit() first.")

        # X Transform
        X_curr = self._apply_l2(X)
        if self.pca:          X_curr = self.pca.transform(X_curr)
        if self.selector_reg: X_curr = self.selector_reg.transform(X_curr)

        df_X = pd.DataFrame(X_curr, index=X.index)

        # Y Transform
        if y is not None:
            if self.y_scaler_:
                y_sc = self.y_scaler_.transform(y)
                df_y = pd.DataFrame(y_sc, index=y.index, columns=y.columns)
            else:
                df_y = y
            return df_X, df_y

        return df_X

    def inverse_transform_y(self, y_scaled):
        if self.y_scaler_ is None:
            return y_scaled

        val = y_scaled.values if isinstance(y_scaled, pd.DataFrame) else y_scaled
        y_inv = self.y_scaler_.inverse_transform(val)

        if isinstance(y_scaled, pd.DataFrame):
            return pd.DataFrame(y_inv, index=y_scaled.index, columns=y_scaled.columns)
        return y_inv

    def _apply_l2(self, X):
        if self.l2_normalize:
            norm = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
            norm[norm == 0] = 1
            return X / norm
        return X
