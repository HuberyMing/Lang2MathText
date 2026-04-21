"""
fMRI DataModule classes.

FMRIDataModule
    Canonical (no-leakage) version. Takes raw dataFMRI.whole arrays directly,
    shuffles them once, and aligns embeddings by index.  Used by the nested CV pipeline.

FMRIDataModule_v1_concatenate_uid
    Alternative variant that concatenates pre-split train/test UIDs at setup time.
    Kept for backward compatibility with dm_choice=1 in run_nested_cv.py.
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class FMRIDataModule:
    def __init__(self, df_emb, y_raw, groups, types, seed=42, **kwargs):
        """
        Args:
            df_emb (pd.DataFrame): Full LLM embeddings (index = stimsetid).
            y_raw  (pd.DataFrame): Full fMRI data (from dataFMRI.whole['UID_fMRI']).
            groups (np.array):     Group IDs (from y_raw.index).
            types  (np.array):     Type IDs  (from dataFMRI.whole['UID_WdIt']).
            seed   (int):          Random seed for shuffle.
        """
        self.seed = seed

        # 1. Synchronized shuffle – keeps X/Y/groups/types aligned
        self.full_y_raw, self.full_groups, self.full_types = shuffle(
            y_raw, groups, types, random_state=seed
        )

        # 2. Align embeddings by stimsetid index
        #    loc handles the many-subjects-per-sentence expansion automatically.
        self.full_X_raw = df_emb.loc[self.full_y_raw.index].copy()

        # 3. Sanity checks
        assert len(self.full_X_raw) == len(self.full_y_raw)
        assert len(self.full_y_raw) == len(self.full_groups)
        assert (self.full_X_raw.index == self.full_y_raw.index).all(), \
            "X and Y indices mismatch!"

        print(f"FMRIDataModule Initialized.")
        print(f"  Total Samples:  {len(self.full_y_raw)}")
        print(f"  Unique Groups:  {len(np.unique(self.full_groups))}")
        print(f"  X shape: {self.full_X_raw.shape}")
        print(f"  Y shape: {self.full_y_raw.shape}")

    def setup(self, stage=None):
        # All data prepared in __init__; nothing to do here.
        pass


class FMRIDataModule_v1_concatenate_uid:
    """Concatenate pre-split train/test UID dicts at setup time."""

    def __init__(self, df_emb, uids, groups, types,
                 agg: str = 'collect', word_item: str = 'ALL', seed=42, **kwargs):

        self.df_emb_raw = df_emb
        self.uids = uids
        self.groups = groups
        self.types = types
        self.hparams = {'agg': agg, 'word_item': word_item, 'seed': seed}

        self.full_X_raw   = None
        self.full_y_raw   = None
        self.full_groups  = None
        self.full_types   = None

    def setup(self, stage=None):
        data_stim = self.uids[self.hparams['agg']][self.hparams['word_item']]
        y_train = data_stim['fMRI']['train']
        y_test  = data_stim['fMRI']['test']

        stim_train = data_stim['stimset']['train']
        stim_test  = data_stim['stimset']['test']
        word_train = data_stim['WordItem']['train']
        word_test  = data_stim['WordItem']['test']

        print("Merging Raw Data for CV...")
        self.full_y_raw = pd.concat([y_train, y_test], axis=0)

        stim_full = pd.concat([stim_train, stim_test], axis=0)
        word_full = pd.concat([word_train, word_test], axis=0)

        unique_ids = (stim_full['stimsetid']
                      if 'stimsetid' in stim_full
                      else stim_full.index)
        self.full_X_raw = self.df_emb_raw.loc[unique_ids].copy()

        self.full_groups = (stim_full['stimsetid'].values
                            if 'stimsetid' in stim_full
                            else stim_full.index.values)

        self.full_types = (word_full.iloc[:, 0].values
                           if isinstance(word_full, pd.DataFrame)
                           else word_full.values)

        print(f"Setup Done.  Raw X: {self.full_X_raw.shape},  Raw Y: {self.full_y_raw.shape}")
