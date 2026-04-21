#%%


__file__ = "fMRI_data_loader.py"

import os   
import pandas as pd
from os.path import join

import json

import numpy as np

from scipy.stats import f
from numpy.linalg import inv

from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

import sys
import time

# current_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
# utils_Abs_path = os.path.abspath('./')
# if utils_Abs_path not in sys.path:
#     sys.path.append(utils_Abs_path)

# sys.path.append('.')
# print(utils_Abs_path)
print(sys.path)

# from . import FMRI_key_list     # (OK) under_src:  python -m data.fMRI_data_loader
                                # (OK) under_src:  exec(open('preprocess.py').read())
                                #                  python preprocess.py 
                                # (ImportError)    exec(open('data/fMRI_data_loader.py').read())
                                #                  python data/fMRI_data_loader.py 
                                # (ImportError) under_data: python fMRI_data_loader.py 

from data import FMRI_key_list  # (OK) under_src:   python -m data.fMRI_data_loader 
                                #                   exec(open('data/fMRI_data_loader.py').read())
                                #   (ModuleNotFoundError): python data/fMRI_data_loader.py 
                                #  

# --------  method 1: Importing from utils directory --------



Choice = 'under_src'
if Choice == 'under_src':
    #   (OK)    python -m data.fMRI_data_loader
    #   (OK)    exec(open('data/fMRI_data_loader.py').read())    
    #   ModuleNotFoundError: python data/fMRI_data_loader.py
    #
    from utils.helper import load_config    # python -m data.fMRI_data_loader
    from utils.Plt_compare import Plt_2ROI_compare, Plt_ROI_performance

    ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '..'))

elif Choice == 'under_data':
    #   (OK)    python fMRI_data_loader.py 
    #   (OK)    python -m fMRI_data_loader
    #   (OK)    exec(open('fMRI_data_loader.py').read())

    # 添加 utils 目錄到路徑
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    utils_Abs_path = os.path.abspath(utils_path)
    if utils_Abs_path not in sys.path:
        sys.path.append(utils_Abs_path)
    print(sys.path)

    from helper import load_config      # at directory utils
    from Plt_compare import Plt_2ROI_compare, Plt_ROI_performance
    from Plt_compare import Plt_select_ROI_Wd_C01

    ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '../..'))


elif Choice == 'under_Prj':
    #   (OK) python -m src.data.fMRI_data_loader
    #   (OK) exec(open('src/data/fMRI_data_loader.py').read())

    # from ..utils.helper import load_config  # (OK1) python -m src.data.fMRI_data_loader
    from src.utils.helper import load_config  # (OK2) python -m src.data.fMRI_data_loader    

    ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '.'))

else:       # not checked
    ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '../..'))

DATAROOT	= join(ROOTDIR, 'data')



#%%


fMRI_C_key = ['C0', 'C1']               #['fMRI_C0', 'fMRI_C1']
stim_C_key = ['C0', 'C1']               #['C0stim', 'C1stim']

class PreprocessFMRI:
    def __init__(self, **kwargs):
    
        FileDir    = kwargs.get('FileDir', 'raw/')
        BrainFile  = kwargs.get('BrainFile', 'BNL_WPR_brain_behavior.csv') 
        stimsetid  = kwargs.get('stimsetid', 'brain-MD1')

        self.stimsetid = stimsetid

        self.raw_data = self.df_stimset_data(FileDir, BrainFile)

        self.FMRI_key_list = self.raw_data['num_columns'][14:]   # from 'Lang1_LH_IFGorb_-47_27_-4' to the end

        # --------------------------------------------- #
        #   get the id_map
        # --------------------------------------------- #
        df_csv  = self.raw_data['csv_raw']

        y_scanid = df_csv['Scanid']
        ALL_UIDs = y_scanid.unique()

        UID_dict = {}
        UID_inv_dict = {}
        for ii, UID in enumerate(ALL_UIDs):
            UID_dict[ii] = UID
            UID_inv_dict[UID] = ii

        y_Uid = [UID_inv_dict[xx] for xx in y_scanid]
        y_Uid = pd.Series(y_Uid, name='subj_id')

        # stimset id ------------------------- 
        y_stimID = df_csv['items_num']
        raw_ind = [stimsetid+'.'+str(ii) for ii in y_stimID]

        #
        #   id_mapping
        #
        id_map = pd.concat((y_stimID, y_Uid), axis=1)
        id_map['stimset'] = raw_ind

        self.map = {
            'id_map': id_map,
            'UID_dict': UID_dict,
            'UID_inv_dict': UID_inv_dict,
            'y_Uid': y_Uid,
            'y_scanid': y_scanid,
            'y_stimID': y_stimID,
            'WordItem': df_csv['WordItem']
        }

    def get_UIDs_info(self, UIDs = [], Uindex='U0'):

        if UIDs == []:
            UIDs = self.raw_data['UIDs_ALL']

        self.UIDs   = UIDs
        self.Uindex = Uindex        # the index referring to UIDs
        stimsetid = self.stimsetid

        # --------------------------------------------- #
        #   get the unique stimset for the given UIDs   #
        # --------------------------------------------- #
        self.unique_UID_stimset = self.df_data_query_UIDs_2_C01(UIDs, stimsetid)

        # ------------------------- #
        #   collect all UIDs info   #
        # ------------------------- #
        self.df_data_UIDs = self.UIDs2_data_into_dict(UIDs, stimsetid)

        self.whole = self.each_UID_data_2_whole_C01(UIDs, fMRI_C_key, stim_C_key)

    # ================================================================= #
    #       New / Modified Functions for Cross Validation               #
    # ================================================================= #

    def create_CV_splits(self, n_splits=5, method='kfold', seed=42):
        """
        預先產生 Cross Validation 的索引。
        針對 C0 和 C1 分別做 Split (Stratified 的概念)，確保每個 Fold 裡 C0/C1 比例固定。
        """
        self.cv_splits_indices = {'C0': [], 'C1': []}
        self.n_splits = n_splits
        self.cv_method = method

        stimset_C01 = self.unique_UID_stimset # {'C0': df, 'C1': df}

        for C_key in stimset_C01.keys(): # Loop C0, C1
            data = stimset_C01[C_key]
            indices = np.arange(data.shape[0])

            if method == 'loocv':
                cv = LeaveOneOut()
                print(f"[{C_key}] Using Leave-One-Out CV (Total {len(indices)} items)")
                self.n_splits = len(indices) # Update n_splits to match data length
            else:
                # Default to K-Fold
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                print(f"[{C_key}] Using {n_splits}-Fold CV")

            # 儲存該類別 (C0 or C1) 所有的 (train_idx, test_idx)
            splits = []
            for train_idx_num, test_idx_num in cv.split(indices):
                # 將數值 Index 轉回 dataframe 的 index (stimsetid)
                train_index = data.iloc[train_idx_num].index
                test_index  = data.iloc[test_idx_num].index
                splits.append((train_index, test_index))
            
            self.cv_splits_indices[C_key] = splits

        # 檢查 C0 和 C1 的 fold 數是否一致 (若是 LOOCV 且題目數不同，這裡會取最小集)
        min_folds = min(len(self.cv_splits_indices['C0']), len(self.cv_splits_indices['C1']))
        if self.n_splits != min_folds and method == 'loocv':
            print(f"Warning: LOOCV item counts mismatch between C0/C1. Using {min_folds} folds.")
            self.n_splits = min_folds

    def load_fold_data(self, fold_index):
        """
        將內部的資料狀態 (self.UIDs_train_test) 切換到指定的 fold_index。
        """
        if not hasattr(self, 'cv_splits_indices'):
            raise ValueError("Please run .create_CV_splits() first.")
        
        if fold_index >= self.n_splits:
            raise ValueError(f"Fold index {fold_index} out of range (Total {self.n_splits} folds).")

        UIDs = self.UIDs
        
        # 1. 建構該 Fold 的 index_train_test_C01
        index_train_C01 = {}
        index_test_C01  = {}

        for C_key in ['C0', 'C1']:
            train_index, test_index = self.cv_splits_indices[C_key][fold_index]
            index_train_C01[C_key] = train_index
            index_test_C01[C_key]  = test_index

        self.index_train_test_C01 = {
            'train': index_train_C01,
            'test': index_test_C01
        }

        # 2. 重用舊有的邏輯來收集與合併資料
        #    因為 self.index_train_test_C01 已經被替換成當前 Fold 的索引，
        #    所以以下的函數會自動抓取對應的資料。
        
        # Collect Raw Data based on indices
        C01_train_test = self.UID_data_ind_to_C01_train_test(UIDs, fMRI_C_key, stim_C_key)
        
        # Combine (Stacking)
        self.UIDs_train_test = self.combine_C01_fMRI_stim(UIDs, C01_train_test)
        
        # Calculate Average
        self.Avg_UIDs_train_test(C01_train_test)

        print(f"    --> Data loaded for Fold {fold_index + 1}/{self.n_splits}")


    # ------------------------------------------------------- #
    # 原本的 Single Split 邏輯
    # ... (select_stimset_train_test 保留原樣或用於單次分割) ...
    # ... (其餘 helper function 如 combine_C01_fMRI_stim, Avg_UIDs_train_test 等皆不需修改) ...
    #
    # ------------------------------------------------------- #
    def select_stimset_train_test(self, test_size=0.2, seed=42):

        UIDs = self.UIDs
        # ------------------------------------------- #
        #   split (train, test) according to stimset  #
        # ------------------------------------------- #
        self.test_size = test_size
        self.seed = seed

        self.index_train_test_C01 = self.each_C01_split_train_test(test_size=test_size, seed=seed)
        
        # ----------------------------- #
        #   collect all over UIDs       #
        # ----------------------------- #
        C01_train_test = self.UID_data_ind_to_C01_train_test(UIDs, fMRI_C_key, stim_C_key)
        # self.C01_train_test = C01_train_test

        self.UIDs_train_test = self.combine_C01_fMRI_stim(UIDs, C01_train_test)

        # ----------------------------- #
        #   get the average over UIDs   #
        # ----------------------------- #
        self.Avg_UIDs_train_test(C01_train_test)

    # --------------------------------------------------- #
    #       average
    # --------------------------------------------------- #

    def Avg_UIDs_fMRI_stimset(self):
        
        self.Avg = dict()
        self.Avg['UIDs'] = self.whole['UIDs']
        self.Avg['stim'] = self.Avg_data_by_stimsetid(self.whole['UID_stim'], 'first')
        self.Avg['fMRI'] = self.Avg_data_by_stimsetid(self.whole['UID_fMRI'], 'mean')


        self.Avg['stim_C01'] = dict()
        self.Avg['fMRI_C01'] = dict()
        for key_st, key_fM in zip(stim_C_key, fMRI_C_key):

            self.Avg['stim_C01'][key_st] = self.Avg_data_by_stimsetid(self.whole['UID_stim_C01'][key_st], 'first')
            self.Avg['fMRI_C01'][key_fM] = self.Avg_data_by_stimsetid(self.whole['UID_fMRI_C01'][key_fM], 'mean')

            assert (self.Avg['stim_C01'][key_st].index == self.Avg['fMRI_C01'][key_fM].index).all()

    def Avg_UIDs_train_test(self, C01_train_test):

        Avg_c0_fMRI_train = self.Avg_data_by_stimsetid(C01_train_test['C0']['fMRI']['train'], 'mean')
        Avg_c0_fMRI_test = self.Avg_data_by_stimsetid(C01_train_test['C0']['fMRI']['test'], 'mean')
        Avg_c1_fMRI_train = self.Avg_data_by_stimsetid(C01_train_test['C1']['fMRI']['train'], 'mean')
        Avg_c1_fMRI_test = self.Avg_data_by_stimsetid(C01_train_test['C1']['fMRI']['test'], 'mean')

        first_c0_stim_train = self.Avg_data_by_stimsetid(C01_train_test['C0']['stimset']['train'], 'first')
        first_c0_stim_test  = self.Avg_data_by_stimsetid(C01_train_test['C0']['stimset']['test'], 'first')
        first_c1_stim_train = self.Avg_data_by_stimsetid(C01_train_test['C1']['stimset']['train'], 'first')
        first_c1_stim_test  = self.Avg_data_by_stimsetid(C01_train_test['C1']['stimset']['test'], 'first')

        assert (Avg_c0_fMRI_train.index == first_c0_stim_train.index).all()
        assert (Avg_c0_fMRI_test.index == first_c0_stim_test.index).all()
        assert (Avg_c1_fMRI_train.index == first_c1_stim_train.index).all()
        assert (Avg_c1_fMRI_test.index == first_c1_stim_test.index).all()

        c0_train = pd.DataFrame(np.zeros((Avg_c0_fMRI_train.shape[0], 1), dtype='int'), 
                                index = Avg_c0_fMRI_train.index)
        c1_train = pd.DataFrame(np.ones((Avg_c1_fMRI_train.shape[0], 1), dtype='int'), 
                                index = Avg_c1_fMRI_train.index)

        c0_test = pd.DataFrame(np.zeros((Avg_c0_fMRI_test.shape[0], 1), dtype='int'), 
                                index = Avg_c0_fMRI_test.index)
        c1_test = pd.DataFrame(np.ones((Avg_c1_fMRI_test.shape[0], 1), dtype='int'), 
                                index = Avg_c1_fMRI_test.index)

        UIDs_Avg = {
            'C0': {
                'fMRI': {
                    'train': Avg_c0_fMRI_train,
                    'test': Avg_c0_fMRI_test
                },
                'WordItem': {
                    'train': c0_train,
                    'test': c0_test
                },
                'stimset': {
                    'train': first_c0_stim_train,
                    'test': first_c0_stim_test
                }
            },
            'C1': {
                'fMRI': {
                    'train': Avg_c1_fMRI_train,
                    'test': Avg_c1_fMRI_test
                },
                'WordItem': {
                    'train': c1_train,
                    'test': c1_test
                },
                'stimset': {
                    'train': first_c1_stim_train,
                    'test': first_c1_stim_test
                }
            },
            'ALL': {
                'fMRI': {
                    'train': pd.concat([Avg_c0_fMRI_train, Avg_c1_fMRI_train], axis=0),
                    'test': pd.concat([Avg_c0_fMRI_test, Avg_c1_fMRI_test], axis=0)
                },
                'WordItem': {
                    'train': pd.concat([c0_train, c1_train], axis=0),
                    'test': pd.concat([c0_test, c1_test], axis=0)
                },
                'stimset': {
                    'train': pd.concat([first_c0_stim_train, first_c1_stim_train], axis=0),
                    'test': pd.concat([first_c0_stim_test, first_c1_stim_test], axis=0)
                }
            }
        }
        assert (UIDs_Avg['ALL']['fMRI']['train'].index == UIDs_Avg['ALL']['WordItem']['train'].index).all()
        assert (UIDs_Avg['ALL']['fMRI']['test'].index == UIDs_Avg['ALL']['WordItem']['test'].index).all()

        self.UIDs_train_test.update({'Avg': UIDs_Avg})


    def Avg_data_by_stimsetid(self, data_given, #key0, key1, key2, 
                            agg='mean'):
        # if agg == 'mean':
        #     Avg = data_dict[key0][key1][key2].groupby('stimsetid').mean()
        # elif agg == 'first':
        #     Avg = data_dict[key0][key1][key2].groupby('stimsetid').first()

        if agg == 'mean':
            Avg = data_given.groupby('stimsetid').mean()
        elif agg == 'first':
            Avg = data_given.groupby('stimsetid').first()
        else:
            print(' Not defined')
        return Avg


    def Compare_Check_Avg_ALL_train_test(self):
        Avg_stim = pd.DataFrame()
        Avg_fMRI = pd.DataFrame()
        C0_fMRI  = pd.DataFrame()
        C0_stim  = pd.DataFrame()
        C1_fMRI  = pd.DataFrame()
        C1_stim  = pd.DataFrame()

        for sp_set in ['train', 'test']:
            Avg_fMRI = pd.concat([Avg_fMRI, self.UIDs_train_test['Avg']['ALL']['fMRI'][sp_set] ])
            Avg_stim = pd.concat([Avg_stim, self.UIDs_train_test['Avg']['ALL']['stimset'][sp_set] ])

            C0_fMRI  = pd.concat([C0_fMRI, self.UIDs_train_test['Avg']['C0']['fMRI'][sp_set] ])
            C0_stim  = pd.concat([C0_stim, self.UIDs_train_test['Avg']['C0']['stimset'][sp_set] ])

            C1_fMRI  = pd.concat([C1_fMRI, self.UIDs_train_test['Avg']['C1']['fMRI'][sp_set] ])
            C1_stim  = pd.concat([C1_stim, self.UIDs_train_test['Avg']['C1']['stimset'][sp_set] ])

        Avg_stim.sort_values(by='stimsetid', inplace=True)
        Avg_fMRI.sort_values(by='stimsetid', inplace=True)
        C0_stim.sort_values(by='stimsetid', inplace=True)
        C0_fMRI.sort_values(by='stimsetid', inplace=True)
        C1_stim.sort_values(by='stimsetid', inplace=True)
        C1_fMRI.sort_values(by='stimsetid', inplace=True)

        assert (C0_stim.index == self.Avg['stim_C01']['C0'].index).all()
        assert (C0_stim == self.Avg['stim_C01']['C0']).all().all()
        assert (C0_fMRI == self.Avg['fMRI_C01']['C0']).all().all()
        assert (C1_stim == self.Avg['stim_C01']['C1']).all().all()
        assert (C1_fMRI == self.Avg['fMRI_C01']['C1']).all().all()

        assert (self.Avg['stim'].index == Avg_stim.index).all()
        assert (self.Avg['stim'] == Avg_stim).all().all()
        assert (self.Avg['fMRI'] == Avg_fMRI).all().all()

        print(' =======  PASS all comparison  =======')


    #
    #   combine (C0, C1) --> ALL
    #
    def combine_C01_fMRI_stim(self, UIDs, C01_train_test):

        UIDs_train_test = {'UIDs': UIDs,
            'collect': C01_train_test      
            }
        UIDs_train_test['collect'].update({
            'ALL': {
                'fMRI': {
                    'train': self.combine_C01(C01_train_test, 'fMRI', 'train'),
                    'test': self.combine_C01(C01_train_test, 'fMRI', 'test')
                },
                'WordItem': {
                    'train': self.combine_C01(C01_train_test, 'fMRI', 'train', 1),
                    'test': self.combine_C01(C01_train_test, 'fMRI', 'test', 1)
                },
                'stimset':{
                    'train': self.combine_C01(C01_train_test, 'stimset', 'train'),
                    'test': self.combine_C01(C01_train_test, 'stimset', 'test')
                },
                'subj':{
                    'train': self.combine_C01(C01_train_test, 'subj', 'train'),
                    'test': self.combine_C01(C01_train_test, 'subj', 'test')                
                }
            }
        })

        assert (UIDs_train_test['collect']['ALL']['fMRI']['train'].index == UIDs_train_test['collect']['ALL']['WordItem']['train'].index).all()
        assert (UIDs_train_test['collect']['ALL']['fMRI']['test'].index == UIDs_train_test['collect']['ALL']['WordItem']['test'].index).all()

        return UIDs_train_test

    def combine_C01(self, data_dict, key1, key2, Option=0):
        combined = pd.DataFrame() 
        for key in data_dict.keys():

            if Option == 0:
                combined = pd.concat([combined, data_dict[key][key1][key2]], axis=0)

            elif Option == 1:
                shape = data_dict[key][key1][key2].shape
                index = data_dict[key][key1][key2].index

                if key == 'C0':
                    Word = pd.DataFrame(np.zeros((shape[0], 1), dtype='int'), index=index)
                elif key == 'C1':
                    Word = pd.DataFrame(np.ones((shape[0], 1), dtype='int'), index=index)
                combined = pd.concat([combined, Word])

        return combined
    

    def combine_C01_train_test(self, fMRI_C_key, stim_C_key):
        """ self.train_test_C01  not implemented yet
            same as self.C01_train_test  but with different dict order
        """
        train_fMRI_C01    = self.train_test_C01['train']['fMRI_C01']
        test_fMRI_C01     = self.train_test_C01['test']['fMRI_C01']
        train_stimset_C01 = self.train_test_C01['train']['stimset_C01']
        test_stimset_C01  = self.train_test_C01['test']['stimset_C01']

        #
        #   combine each (C0, C1) train & test --> into the whole train & tests
        #
        train_fMRI     = pd.DataFrame()     # only fMRI columns
        test_fMRI      = pd.DataFrame()     # only fMRI columns
        train_stimset  = pd.DataFrame()     # only stimset columns
        test_stimset   = pd.DataFrame()     # only stimset columns

        for C_fMRI, C_stim in zip(fMRI_C_key, stim_C_key):

            train_fMRI   = pd.concat([train_fMRI, train_fMRI_C01[C_fMRI]], axis=0)
            test_fMRI    = pd.concat([test_fMRI, test_fMRI_C01[C_fMRI]], axis=0)

            train_stimset = pd.concat([train_stimset, train_stimset_C01[C_stim]], axis=0) 
            test_stimset  = pd.concat([test_stimset, test_stimset_C01[C_stim]], axis=0)

            assert (train_stimset.index == train_fMRI.index).all()
            assert (test_stimset.index == test_fMRI.index).all()

        return train_fMRI, test_fMRI, train_stimset, test_stimset

    #
    #   determine the train & test index
    #
    def each_C01_split_train_test(self, test_size=0.2, seed=42):

        stimset_C01 = self.unique_UID_stimset

        index_train_C01 = dict()
        index_test_C01  = dict()
        for C_stim in stimset_C01.keys():

            index_train, index_test = self.split_InputData_train_test(stimset_C01[C_stim], test_size=test_size, seed=seed)

            index_train_C01[C_stim] = index_train
            index_test_C01[C_stim]  = index_test

        index_train_test_C01 = {
            'train': index_train_C01, 
            'test': index_test_C01
        }

        return index_train_test_C01


    def split_InputData_train_test(self, input_data, test_size=0.2, seed=0):

        #
        #   method 1: self-defined function train_test_split_ID
        #
        # train_idx, test_idx = train_test_split_ID(input_data, test_size=0.2, random_state=seed)

        #
        #   method 2: sklearn.model_selection.train_test_split
        #
        train_idx, test_idx = train_test_split(np.arange(input_data.shape[0]), test_size=test_size, random_state=seed)
        assert (np.sort(np.concatenate((train_idx, test_idx))) == np.arange(input_data.shape[0])).all()

        index_train = input_data.iloc[train_idx].index
        index_test  = input_data.iloc[test_idx].index
        assert sorted(input_data.index) == sorted(np.concatenate([index_train, index_test]))
        
        return index_train, index_test


    #
    #   collect over ALL UIDs
    #
    def UID_data_ind_to_C01_train_test(self, UIDs, fMRI_C_key, stim_C_key):
        """ from the index (index_train_C01, index_test_C01)
            to obtain the real fMRI & stimset data

        Args:
            df_data_UIDs (dict): store each UID data ('neural_fMRI', 'stimset', 'C0', 'C1', 'C0stim', 'C1stim')
            index_train_C01 (_type_): the train index for each C_stim \in stim_C_key = ['C0stim', 'C1stim']
            index_test_C01 (_type_):  the  test index for each C_stim \in stim_C_key = ['C0stim', 'C1stim']
            fMRI_C_key (list): fMRI_C_key = ['C0', 'C1']            # ['fMRI_C0', 'fMRI_C1']
            stim_C_key (list): stim_C_key = ['C0stim', 'C1stim']

        return the (train & test) for real fMRI & stimset data
                    for each Cond, i.e.  stim_C_key = ['C0stim', 'C1stim']
        """
        #
        #   split data according to the whole_UID_stim_C01 (stimset for each C0 or C1) 
        #       --> find the corresponding indices (index_train_C01, index_test_C01)
        #                   in each UID data as the input
        #       --> each UID data split into train and test for each C0 or C1
        #

        #
        #   have the train & test for each C0 & C1
        #
        train_fMRI_C01  = {
            'C0': pd.DataFrame(), # collect train fMRI for C0
            'C1': pd.DataFrame(), # collect train fMRI for C1
        }
        test_fMRI_C01  = {
            'C0': pd.DataFrame(), # collect test fMRI for C0
            'C1': pd.DataFrame(), # collect test fMRI for C1
        }

        train_stimset_C01  = {
            'C0': pd.DataFrame(), # collect train stimset for C0
            'C1': pd.DataFrame(), # collect train stimset for C1
        }
        test_stimset_C01  = {
            'C0': pd.DataFrame(), # collect test stimset for C0
            'C1': pd.DataFrame(), # collect test stimset for C1
        }

        train_subj_C01  = {
            'C0': pd.DataFrame(), # collect train subjectID for C0
            'C1': pd.DataFrame(), # collect train subjectID for C1
        }
        test_subj_C01  = {
            'C0': pd.DataFrame(), # collect train subjectID for C0
            'C1': pd.DataFrame(), # collect train subjectID for C1
        }


        for UID in UIDs:
            
            subj_id = self.df_data_UIDs[UID]['subj_id']
            assert UID == self.map['UID_dict'][subj_id]

            # for C_fMRI, C_stim in zip(fMRI_C_key, stim_C_key):
            for UID_C_fMRI, UID_C_stim, C_fMRI, C_stim in zip(['fMRI_C0', 'fMRI_C1'], ['stim_C0', 'stim_C1'], fMRI_C_key, stim_C_key):

                each_UID_select_fMRI = self.df_data_UIDs[UID][UID_C_fMRI]    # UID_C_fMRI \in ['fMRI_C0', 'fMRI_C1']
                each_UID_select_stim = self.df_data_UIDs[UID][UID_C_stim]    # UID_C_stim \in ['stim_C0', 'stim_C1']



                index_train = self.index_train_test_C01['train'][C_stim]
                index_test  = self.index_train_test_C01['test'][C_stim]

                common_train = list(set(each_UID_select_stim.index) & set(index_train))
                common_test  = list(set(each_UID_select_stim.index) & set(index_test))

                train_Inp_fMRI, test_Inp_fMRI, train_Inp_stimset, test_Inp_stimset = \
                    self.split_fMRI_stimset_train_test_by_index(each_UID_select_fMRI, each_UID_select_stim, common_train, common_test)

                # train_Inp_fMRI, test_Inp_fMRI, train_Inp_stimset, test_Inp_stimset = \
                #     split_fMRI_stimset_train_test_by_index(each_UID_select_fMRI, each_UID_select_stim, index_train, index_test)

                # print(f'UID = {UID}, C_fMRI = {C_fMRI}, C_stim = {C_stim}')
                # print(f'  index_test = {index_test}\n')
                # print(f'  test_Inp_stimset.index = {test_Inp_stimset.index}')
                # print(f'  test_Inp_fMRI.index    = {test_Inp_fMRI.index}\n\n')

                train_fMRI_C01[C_fMRI] = pd.concat([train_fMRI_C01[C_fMRI], train_Inp_fMRI], axis=0)
                test_fMRI_C01[C_fMRI]  = pd.concat([test_fMRI_C01[C_fMRI], test_Inp_fMRI], axis=0)

                train_stimset_C01[C_stim] = pd.concat([train_stimset_C01[C_stim], train_Inp_stimset], axis=0)
                test_stimset_C01[C_stim] = pd.concat([test_stimset_C01[C_stim], test_Inp_stimset], axis=0)

                assert (train_stimset_C01[C_stim].index == train_fMRI_C01[C_fMRI].index).all()  # train set
                assert (test_stimset_C01[C_stim].index == test_fMRI_C01[C_fMRI].index).all()    # test set

                N_UID_train = train_Inp_stimset.shape[0]
                N_UID_test  = test_Inp_stimset.shape[0]
                train_UID_subj = pd.DataFrame(np.ones(N_UID_train, dtype='int')* subj_id)
                test_UID_subj  = pd.DataFrame(np.ones(N_UID_test,  dtype='int')* subj_id)
                train_subj_C01[C_stim] = pd.concat([train_subj_C01[C_stim], train_UID_subj], axis=0)
                test_subj_C01[C_stim]  = pd.concat([test_subj_C01[C_stim],  test_UID_subj],  axis=0)


        # train_test_C01 = {
        #     'train_fMRI_C01': train_fMRI_C01, 
        #     'test_fMRI_C01': test_fMRI_C01, 
        #     'train_stimset_C01': train_stimset_C01, 
        #     'test_stimset_C01': test_stimset_C01
        # }

        # train_test_C01 = {
        #     'train': {
        #         'fMRI_C01': train_fMRI_C01,
        #         'stimset_C01': train_stimset_C01
        #     },
        #     'test': {
        #         'fMRI_C01': test_fMRI_C01,
        #         'stimset_C01': test_stimset_C01
        #     }
        # }

        c0_train = pd.DataFrame(np.zeros((train_fMRI_C01['C0'].shape[0], 1), dtype='int'), 
                                index = train_fMRI_C01['C0'].index)
        c1_train = pd.DataFrame(np.ones((train_fMRI_C01['C1'].shape[0], 1), dtype='int'), 
                                index = train_fMRI_C01['C1'].index)

        c0_test = pd.DataFrame(np.zeros((test_fMRI_C01['C0'].shape[0], 1), dtype='int'), 
                                index = test_fMRI_C01['C0'].index)
        c1_test = pd.DataFrame(np.ones((test_fMRI_C01['C1'].shape[0], 1), dtype='int'), 
                                index = test_fMRI_C01['C1'].index)

        C01_train_test = {
            'C0':{
                'fMRI': {
                    'train': train_fMRI_C01['C0'],
                    'test': test_fMRI_C01['C0']
                },
                'WordItem': {
                    'train': c0_train,
                    'test': c0_test
                },
                'stimset': {
                    'train': train_stimset_C01['C0'],
                    'test': test_stimset_C01['C0']
                },
                'subj': {
                    'train': train_subj_C01['C0'],
                    'test': test_subj_C01['C0']
                }
            },
            'C1':{
                'fMRI': {
                    'train': train_fMRI_C01['C1'],
                    'test': test_fMRI_C01['C1']
                },
                'WordItem': {
                    'train': c1_train,
                    'test': c1_test
                },
                'stimset': {
                    'train': train_stimset_C01['C1'],
                    'test': test_stimset_C01['C1']
                },
                'subj': {
                    'train': train_subj_C01['C1'],
                    'test': test_subj_C01['C1']
                }
            }
        }

        return C01_train_test


    def split_fMRI_stimset_train_test_by_index(self, each_UID_select_fMRI, each_UID_select_stim, index_train, index_test):
        """ split according to index_train, index_test,
                i.e. the index name

            must guarantee (index_train, index_test) \in each_UID_select_fMRI.index
                                                i.e. \in each_UID_select_stim.index
            --> therefore, (index_train, index_test) must be given by 
                        (common_train, common_test) from outside
        """

        assert (np.sort(np.concatenate((index_train, index_test))) == sorted(each_UID_select_fMRI.index)).all()
        assert (each_UID_select_fMRI.index == each_UID_select_stim.index).all()
        #
        #   each_UID_select_stim has column ['item_id'] 
        #
        index_ID = [int(x.split('.')[1]) for x in each_UID_select_stim.index]
        assert (each_UID_select_stim['item_id'] == index_ID).all()

        train_Inp_fMRI = each_UID_select_fMRI.loc[index_train, :]#.reset_index(drop=True)
        test_Inp_fMRI  = each_UID_select_fMRI.loc[index_test, :]#.reset_index

        train_Inp_stimset = each_UID_select_stim.loc[index_train, :]#.reset_index(drop=True)
        test_Inp_stimset  = each_UID_select_stim.loc[index_test, :]#.reset_index(drop=True)

        assert (train_Inp_fMRI.index == train_Inp_stimset.index).all()
        assert (test_Inp_fMRI.index  == test_Inp_stimset.index).all()
        assert sorted(each_UID_select_fMRI.index) == sorted(np.concatenate([train_Inp_fMRI.index.values, test_Inp_fMRI.index.values]))

        return train_Inp_fMRI, test_Inp_fMRI, train_Inp_stimset, test_Inp_stimset



    def each_UID_data_2_whole_C01(self, UIDs, fMRI_C_key, stim_C_key):
        """ collect the whole data of the selected UIDs
            and prepare for cross-validation
        """

        df_data_UIDs    = self.df_data_UIDs

        whole_UID_fMRI  = pd.DataFrame()     # include all columns
        whole_UID_stim  = pd.DataFrame()
        whole_UID_subj  = pd.DataFrame()     # subject ID (Uid)

        whole_UID_stim_C01 = {
            'C0': pd.DataFrame(), # collect stimset for C0
            'C1': pd.DataFrame(), # collect stimset for C1
        }
        whole_UID_fMRI_C01 = {
            'C0': pd.DataFrame(), # collect fMRI for C0
            'C1': pd.DataFrame(), # collect fMRI for C1
        }
        whole_UID_subj_C01 = {
            'C0': pd.DataFrame(), # collect subj for C0
            'C1': pd.DataFrame(), # collect subj for C1
        }


        for UID in UIDs:
            whole_UID_fMRI = pd.concat([whole_UID_fMRI, df_data_UIDs[UID]['neural_fMRI']], axis=0)
            whole_UID_stim = pd.concat([whole_UID_stim, df_data_UIDs[UID]['stimset']], axis=0)

            subj_id = df_data_UIDs[UID]['subj_id']
            assert UID == self.map['UID_dict'][subj_id]
            N_stim = df_data_UIDs[UID]['stimset'].shape[0]
            whole_UID_subj = pd.concat([whole_UID_subj, 
                                pd.DataFrame(np.ones(N_stim, dtype='int')* subj_id) ], axis=0)

            assert set(df_data_UIDs[UID]['fMRI_C0'].index).issubset(set(whole_UID_fMRI.index))
            assert set(df_data_UIDs[UID]['fMRI_C1'].index).issubset(set(whole_UID_fMRI.index))
            
            # for C_fMRI, C_stim in zip(fMRI_C_key, stim_C_key):
            for UID_C_fMRI, UID_C_stim, C_fMRI, C_stim in zip(['fMRI_C0', 'fMRI_C1'], ['stim_C0', 'stim_C1'], fMRI_C_key, stim_C_key):

                each_UID_select_fMRI = df_data_UIDs[UID][UID_C_fMRI]    # C_fMRI \in ['fMRI_C0', 'fMRI_C1']
                each_UID_select_stim = df_data_UIDs[UID][UID_C_stim]    # C_stim \in ['stim_C0', 'stim_C1']

                N_select_C01_stim = each_UID_select_stim.shape[0]
                each_UID_select_subj = pd.DataFrame(np.ones(N_select_C01_stim, dtype='int')* subj_id)

                #
                #   stimset data for each C_stim (C0 or C1)
                #       --> drop the repeated
                #
                whole_UID_stim_C01[C_stim] = pd.concat([whole_UID_stim_C01[C_stim], each_UID_select_stim], axis=0)
                whole_UID_fMRI_C01[C_fMRI] = pd.concat([whole_UID_fMRI_C01[C_fMRI], each_UID_select_fMRI], axis=0)

                whole_UID_subj_C01[C_stim] = pd.concat([whole_UID_subj_C01[C_stim], each_UID_select_subj], axis=0)
        #
        #   only take the non-repeated ones
        #   
        stimset_C01 = {
            'C0': pd.DataFrame(), # collect stimset for C0
            'C1': pd.DataFrame(), # collect stimset for C1
        }
        for C_fMRI, C_stim in zip(fMRI_C_key, stim_C_key):

            # check each sentence has the same item_id
            assert (whole_UID_stim_C01[C_stim].groupby('item_id').first().index == sorted(whole_UID_stim_C01[C_stim].groupby('sentence').mean()['item_id'].values)).all()

            stimset_C01[C_stim] = whole_UID_stim_C01[C_stim].groupby('item_id').first().reset_index()
            stimset_C01[C_stim].index = self.stimsetid + '.' + stimset_C01[C_stim]['item_id'].astype(str)

            stimset_C01[C_stim].index.name = 'stimsetid'

        # whole_UID_fMRI.index = stimsetid + '.' + whole_UID_fMRI['items_num'].astype(str)
        # whole_UID_fMRI.index.name = 'stimsetid'
        # assert set(whole_UID_fMRI.index) == set(df_data_Uavg.index)

        assert (self.unique_UID_stimset['C0'] == stimset_C01['C0']).all().all()
        assert (self.unique_UID_stimset['C1'] == stimset_C01['C1']).all().all()

        whole_UID_subj.index      = whole_UID_stim.index
        # whole_UID_subj.index.name = whole_UID_stim.index.name
        whole_UID_subj.columns    = ['Uid']

        whole = {
            'UIDs':         UIDs,
            'UID_fMRI':     whole_UID_fMRI.loc[:, FMRI_key_list], 
            'UID_stim':     whole_UID_stim,
            'UID_WdIt':     pd.DataFrame(whole_UID_fMRI.loc[:, 'WordItem']), 
            'UID_subj':     pd.DataFrame(whole_UID_subj),
            'UID_fMRI_C01': whole_UID_fMRI_C01, 
            'UID_stim_C01': whole_UID_stim_C01, 
            'UID_subj_C01': whole_UID_subj_C01
            # 'stimset_C01': stimset_C01
        }
        return whole


    def UIDs2_data_into_dict(self, UIDs, stimsetid):
        # ------------------------------------------------------- #
        #               preprocess the df_data                    #
        # ------------------------------------------------------- #

        df_data = self.raw_data['csv_raw']

        df_data_UIDs = dict()

        for UID in UIDs:

            UID_fMRI = df_data.query('Scanid == @UID').reset_index(drop=True)

            fMRI_stimset_C0C1 = self.Input_df_fMRI_stim_get_C0C1(UID, UID_fMRI, stimsetid)

            df_data_UIDs[UID] = fMRI_stimset_C0C1

        return df_data_UIDs


    def Input_df_fMRI_stim_get_C0C1(self, UID, df_Input_data, stimsetid='brain-MD1', Target_col = None):
        """_summary_

        Args:
            df_Input_data (pd.DataFrame): the selected fMRI+stimset data, usually a single UID
                --> need the columns [['items_num', 'items']]
                --> i.e. will be changed to name of ['item_id', 'sentence'] for stimset

            df_data (pd.DataFrame): the whole data 
            stimsetid (str): Default to 'brain-MD1'
            Target_col (list, optional): (eg) FMRI_key_list or None. Defaults to None.

        Returns:
            _type_: _description_
        """
        import copy

        Target_col = None
        # Target_col = FMRI_key_list

        if Target_col is None:
            neural_data   = copy.deepcopy(df_Input_data)
            neural_data.index = stimsetid + '.' + neural_data.items_num.astype(str)
        else:
            # neural_data = df_Input_data[Target_col]  
            NewList     = ['WordItem', 'items_num', 'items'] + Target_col 
            neural_data = df_Input_data[NewList]
            neural_data.index = stimsetid + '.' + neural_data.items_num.astype(str)
            # neural_data = neural_data[Target_col]

        neural_data.index.name = 'stimsetid'

        C0 = neural_data.query('WordItem==0')
        C1 = neural_data.query('WordItem==1')
        assert (pd.concat([C0,C1]).sort_values(by='items_num') == neural_data.sort_values(by='items_num')).all().all()

        #
        #   stimset: to get the corresponding sentences
        #
        stimset = neural_data[['items_num', 'items']]
        stimset.columns = ['item_id', 'sentence']

        assert (df_Input_data.items_num.values == stimset.item_id.values).all()

        #
        #   C0stim, C1stim = the corresponding stimset for C0 and C1
        #   
        C0ind = C0['items_num'].values
        C1ind = C1['items_num'].values
        assert C1ind.shape == C0ind.shape           # check equal number of C0 and C1
        assert C1ind.shape[0] + C0ind.shape[0] == stimset.shape[0]

        C0stim = stimset.query('item_id in @C0ind')
        C1stim = stimset.query('item_id in @C1ind')
        assert (stimset.query('item_id not in @C0ind') == C1stim).all().all()
        assert (pd.concat([C0stim, C1stim]).sort_values(by='item_id') == stimset.sort_values(by='item_id')).all().all()

        fMRI_stimset_C0C1 = {
            'neural_fMRI': neural_data,
            'stimset': stimset,
            # 'stimset_Uniq': stimset_Uniq,
            'fMRI_C0': C0.loc[:, FMRI_key_list],
            'fMRI_C1': C1.loc[:, FMRI_key_list],
            'stim_C0': C0stim,
            'stim_C1': C1stim,
            'UID': UID,
            'subj_id': self.map['UID_inv_dict'][UID]
        }
        assert (fMRI_stimset_C0C1['fMRI_C0'].index == fMRI_stimset_C0C1['stim_C0'].index).all()
        assert (fMRI_stimset_C0C1['fMRI_C1'].index == fMRI_stimset_C0C1['stim_C1'].index).all()

        return fMRI_stimset_C0C1

    #
    #   read in information 
    #
    def df_stimset_data(self, FileDir, BrainFile):

        FilePath = join(DATAROOT, FileDir, BrainFile)
        print(f'DATAROOT = {DATAROOT}')
        print(f'FilePath = {FilePath}')

        df_data = pd.read_csv(FilePath, encoding='Big5')
        print(f'df_data.shape = {df_data.shape}')


        txt_columns = df_data.select_dtypes(include=['object']).columns.tolist()
        num_columns = df_data.select_dtypes(include=['number']).columns.tolist()
        # print(f'txt_columns = {txt_columns}')
        # print(f'num_columns = {num_columns}')

        UIDs_ALL = df_data['Scanid'].unique().tolist()
        # print(f'UIDs_ALL (total {len(UIDs_ALL)}) = {UIDs_ALL}')

        stimset_ALL = df_data[['items_num', 'items']].drop_duplicates().reset_index(drop=True)
        stimset_ALL.columns = ['item_id', 'sentence']

        stimset_ALL.index = self.stimsetid + '.' + stimset_ALL['item_id'].astype(str)
        stimset_ALL.index.name = 'stimsetid'
        stimset_ALL.sort_values(by='item_id', inplace=True)

        print(f'stimset_ALL.shape = {stimset_ALL.shape}')

        raw_data = {'csv_raw': df_data,
            'FileDir': FileDir,
            'BrainFile': BrainFile,
            'txt_columns': txt_columns,
            'num_columns': num_columns,
            'UIDs_ALL': UIDs_ALL,
            'stimset_ALL': stimset_ALL
        }
        return raw_data

    def reset_df_index(self, data: pd.DataFrame, stimsetid='brain-MD1',
                    key: str = 'items_num', 
                    ind_name: str ='stimsetid'):
        
        data = data.reset_index()
        data.index = stimsetid + '.' + data[key].astype(str)
        data.index.name = ind_name
        return data

    def df_data_query_UIDs_2_C01(self, UIDs, stimsetid='brain-MD1'):
        #
        #   deal with (C0, C1)
        #
        df_data = self.raw_data['csv_raw']

        stimset_C0 = df_data.query('Scanid in @UIDs').query('WordItem==0')[['items_num', 'items']].groupby('items_num').first()
        stimset_C1 = df_data.query('Scanid in @UIDs').query('WordItem==1')[['items_num', 'items']].groupby('items_num').first()

        # stimset_C0 = stimset_C0.reset_index()
        # stimset_C0.index = stimsetid + '.' + stimset_C0['items_num'].astype(str)
        # stimset_C0.index.name = 'stimsetid'

        stimset_C0 = self.reset_df_index(stimset_C0, stimsetid=stimsetid, key = 'items_num', ind_name='stimsetid')
        stimset_C0.columns = ['item_id', 'sentence']

        stimset_C1 = self.reset_df_index(stimset_C1, stimsetid=stimsetid, key = 'items_num', ind_name='stimsetid')
        stimset_C1.columns = ['item_id', 'sentence']

        unique_UID_stimset = {
            'C0': stimset_C0,
            'C1': stimset_C1
        }
        
        return unique_UID_stimset


def Load_dataset(
            File,
            seed      = 42,
            test_size = 0.2
        ) -> tuple[PreprocessFMRI, list]:

    config = load_config(File)
    print(config)

    t1 = time.time()

    dataset = PreprocessFMRI(**config)

    FMRI_key_list = dataset.FMRI_key_list       # from 'Lang1_LH_IFGorb_-47_27_-4' to the end

    #
    #   given UIDs, df_data
    #   --> collect all UIDs info
    #

    UIDs = dataset.raw_data['UIDs_ALL']; Uindex = 'U0'
    # UIDs = UIDs_ALL
    # UIDs = ['17-04-10.1', '17-04-10.2'];  # subDir = 'U1'
    # UIDs = ['17-04-10.1', '17-04-10.2', '17-04-10.4']
    # UIDs = ['17-04-10.1', '17-04-10.2', '17-04-10.4', '17-04-11.1']
    # UIDs = ['17-04-10.1', '17-04-11.1']
    # UIDs = ['17-04-10.1', '17-04-11.1', '17-05-20.1', '17-05-20.2']

    dataset.get_UIDs_info(UIDs = UIDs, Uindex=Uindex)
    dataset.Avg_UIDs_fMRI_stimset()

    #
    #   -->  separate (C0, C1) data
    #   -->  split the (train, test) separately
    #
    dataset.select_stimset_train_test(test_size, seed)
    dataset.Compare_Check_Avg_ALL_train_test()

    t2  = time.time()
    print(f'    loading dataset ->  time = {t2-t1}')

    return dataset, FMRI_key_list

# ------------------------------------ #
#      scale transform the data        #
# ------------------------------------ #

def scale_train_test_data(y_train, y_test):

    # 建立 StandardScaler 實例
    y_scaler = StandardScaler()

    # 使用訓練集來擬合 (fit) 標準化器
    # 這一步非常重要，確保只用訓練集的統計量來計算平均值和標準差
    y_train_scaled = y_scaler.fit_transform(y_train)

    # 使用訓練好的標準化器來轉換 (transform) 測試集
    y_test_scaled = y_scaler.transform(y_test)

    # print("\n標準化後 y_train_scaled 的形狀:", y_train_scaled.shape)
    # print("標準化後 y_train_scaled 前五筆資料:\n", y_train_scaled[:5])
    # print("標準化後 y_test_scaled 前五筆資料:\n", y_test_scaled[:5])

    df_y_train_scaled = pd.DataFrame(y_train_scaled, index=y_train.index, columns=y_train.columns)
    df_y_test_scaled = pd.DataFrame(y_test_scaled, index=y_test.index, columns=y_test.columns)

    print(f' ----------- StandardScaler done ----------- ')

    return y_scaler, df_y_train_scaled, df_y_test_scaled


def scale_back_data(y_scaler, df_y_train_scaled, df_y_test_scaled, y_train, y_test):

    #   逆向轉換 (Inverse Transform)
    # 使用 y_scaler 的 inverse_transform 方法進行逆向轉換
    #
    y_train_scBack = y_scaler.inverse_transform(df_y_train_scaled)
    y_test_scBack = y_scaler.inverse_transform(df_y_test_scaled)

    df_y_train_scBack = pd.DataFrame(y_train_scBack, index=y_train.index, columns=y_train.columns)
    df_y_test_scBack = pd.DataFrame(y_test_scBack, index=y_test.index, columns=y_test.columns)

    return df_y_train_scBack, df_y_test_scBack

def Plt_scale_train_test(y_train, df_y_train_scaled, df_y_train_scBack,
                         y_test, df_y_test_scaled, df_y_test_scBack, 
                        features, select_fID):

    from Plt_compare import Plt_2ROI
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))
    Plt_2ROI(df_y_train_scaled, features, select_fID, ax[0,0], leg='train scaled')
    Plt_2ROI(df_y_train_scBack, features, select_fID, ax[0,1], leg='train scBack')
    Plt_2ROI(y_train, features, select_fID, ax[1,0], leg='train')

    Plt_2ROI(df_y_train_scaled, features, select_fID, ax[1,1], leg='train scaled')
    Plt_2ROI(df_y_test_scaled, features, select_fID, ax[1,1], leg='test scaled')


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))
    Plt_2ROI(df_y_test_scaled, features, select_fID, ax[0,0], leg='test scaled')
    Plt_2ROI(df_y_test_scBack, features, select_fID, ax[0,1], leg='test scBack')
    Plt_2ROI(y_test, features, select_fID, ax[1,0], leg='test')

    Plt_2ROI(df_y_train_scBack, features, select_fID, ax[1,1], leg='train scBack')
    Plt_2ROI(df_y_test_scBack, features, select_fID, ax[1,1], leg='test scBack')

    plt.show()

def check_StandardScaler_for_fMRI():
    # y_train = dataset.whole['UID_fMRI']
    # y_test  = dataset.whole['UID_fMRI_C01']['C0']

    # y_train = dataset.UIDs_train_test['Avg']['ALL']['fMRI']['train']
    # y_test  = dataset.UIDs_train_test['Avg']['ALL']['fMRI']['test']    

    y_train = dataset.UIDs_train_test['collect']['ALL']['fMRI']['train']
    y_test  = dataset.UIDs_train_test['collect']['ALL']['fMRI']['test']

    y_scaler, df_y_train_scaled, df_y_test_scaled = scale_train_test_data(y_train, y_test)

    df_y_train_scBack, df_y_test_scBack = \
            scale_back_data(y_scaler, df_y_train_scaled, df_y_test_scaled, y_train, y_test)


    from sklearn.metrics import mean_squared_error, r2_score

    # 評估模型性能
    y_pred = y_test     # check ideal case
    mse_ideal = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2_ideal = r2_score(y_test, y_pred, multioutput='raw_values')

    Plt_scale_train_test(y_train, df_y_train_scaled, df_y_train_scBack,
                         y_test, df_y_test_scaled, df_y_test_scBack, 
                        features, select_fID3)

    Plt_scale_train_test(y_train, df_y_train_scaled, df_y_train_scBack,
                         y_test, df_y_test_scaled, df_y_test_scBack, 
                        features, select_fID2)


def record_Plt_ROI_fMRI_case(Dir_save):
    #
    #   ALL
    #
    Plt_2ROI_compare(dataset.whole['UID_fMRI_C01'], 
            features, select_fID1L, select_fID1R, select_fID2, select_fID3, Dir_save, 'fMRI_ALL', 'fMRI ALL')

    Plt_ROI_performance(dataset.whole['UID_fMRI_C01'], 
                        dataset.whole['UID_fMRI_C01'], 
                features, select_MD, ['fMRI', 'fMRI'], Dir_save, 'MD_ALL_fMRI_fMRI', 0)
    Plt_ROI_performance(dataset.whole['UID_fMRI_C01'], 
                        dataset.whole['UID_fMRI_C01'], 
                features, select_Lang, ['fMRI', 'fMRI'], Dir_save, 'Lang_ALL_fMRI_fMRI', 0)

    #
    #   average
    #
    Plt_2ROI_compare(dataset.Avg['fMRI_C01'], 
            features, select_fID1L, select_fID1R, select_fID2, select_fID3, Dir_save, 'fMRI_avg', 'fMRI avg')

    Plt_ROI_performance(dataset.Avg['fMRI_C01'], 
                        dataset.Avg['fMRI_C01'], 
                    features, select_MD, ['fMRI', 'fMRI'], Dir_save, 'MD_avg_fMRI_fMRI', 1)
    Plt_ROI_performance(dataset.Avg['fMRI_C01'], 
                        dataset.Avg['fMRI_C01'], 
                    features, select_Lang, ['fMRI', 'fMRI'], Dir_save, 'Lang_avg_fMRI_fMRI', 1)



def uv_Rotation(x_data_label, y_data_label):
    (xdata, xlabel) = x_data_label
    (ydata, ylabel) = y_data_label

    u_data = (xdata + ydata) * np.sqrt(1/2)
    v_data = (xdata - ydata) * np.sqrt(1/2)

    r_data = np.sqrt(xdata**2 + ydata**2)


    axNow.scatter(u_data, v_data, label='C0')

    axNow.set_xlabel('u+v')
    axNow.set_ylabel('u-v')

    axNow.legend(loc='upper left', shadow=False, fontsize=7)

    axNow.set_box_aspect(1)
    axNow.set_aspect('equal')
    axNow.grid(True, which='both')

    axNow.axhline(y=0, color='k')
    axNow.axvline(x=0, color='k')

    return u_data, v_data

# --------------------------------- #
#           statistics              #
# --------------------------------- #

from sklearn.preprocessing import StandardScaler





#%%
if __name__ == "__main__":

    if Choice == 'under_src':
        File = "../config/data_config.yaml"    #  from the src directory
    if Choice == 'under_data':
        File = "../../config/data_config.yaml"    #  from the src/data directory
    if Choice == 'under_Prj':
        File = "config/data_config.yaml"        #  from the project root

    print(f' -------------------------------------- ')
    print(f'ROOTDIR  = {ROOTDIR}')
    print(f'DATAROOT = {DATAROOT}')

    config = load_config(File)
    print(config)

    dataset = PreprocessFMRI(**config)

    FMRI_key_list = dataset.FMRI_key_list       # from 'Lang1_LH_IFGorb_-47_27_-4' to the end


    #
    #   select the subject IDs manually
    #

    UIDs = dataset.raw_data['UIDs_ALL']; Uindex = 'U0'
    # UIDs = UIDs_ALL
    # UIDs = ['17-04-10.1', '17-04-10.2'];  # subDir = 'U1'
    # UIDs = ['17-04-10.1', '17-04-10.2', '17-04-10.4']
    # UIDs = ['17-04-10.1', '17-04-10.2', '17-04-10.4', '17-04-11.1']
    # UIDs = ['17-04-10.1', '17-04-11.1']
    # UIDs = ['17-04-10.1', '17-04-11.1', '17-05-20.1', '17-05-20.2']

    
    dataset.get_UIDs_info(UIDs = UIDs, Uindex=Uindex)
    dataset.Avg_UIDs_fMRI_stimset()

    # ------------------------------------------------ #
    #   given UIDs, df_data                            #
    #   --> collect all UIDs info                      #
    #   -->  separate (C0, C1) data                    #
    #   -->  split the (train, test) separately        #
    # ------------------------------------------------ #
    seed      = 42
    test_size = 0.2

    dataset.select_stimset_train_test(test_size, seed)
    dataset.Compare_Check_Avg_ALL_train_test()


    assert (dataset.UIDs_train_test['Avg']['ALL']['fMRI']['train'].index 
            == dataset.UIDs_train_test['Avg']['ALL']['WordItem']['train'].index).all()
    assert (dataset.UIDs_train_test['Avg']['ALL']['fMRI']['test'].index 
            == dataset.UIDs_train_test['Avg']['ALL']['WordItem']['test'].index).all()

    assert (dataset.UIDs_train_test['collect']['ALL']['fMRI']['train'].index == dataset.UIDs_train_test['collect']['ALL']['WordItem']['train'].index).all()
    assert (dataset.UIDs_train_test['collect']['ALL']['fMRI']['test'].index == dataset.UIDs_train_test['collect']['ALL']['WordItem']['test'].index).all()


    #
    #   anlayaze the fMRI data
    #
    def analyze_fMRI_data():
        # fMRI0 = dataset.whole['UID_fMRI_C01']['C0']
        # fMRI1 = dataset.whole['UID_fMRI_C01']['C1']
        # fMRI0 = dataset.Avg['fMRI_C01']['C0']
        # fMRI1 = dataset.Avg['fMRI_C01']['C1']

        features = {id+1: fMRI for id, fMRI in enumerate(FMRI_key_list)}


        select_Lang = [6, 22, 40, 59]   # 'Lang1_LH_netw' 'Lang1_RH_netw' 'Lang2_netw' 'Lang3_netw'
        select_MD   = [16, 32, 52, 68]  # 'MD1_LH_netw' 'MD1_RH_netw' 'MD2_netw' 'MD3_netw'

        select_fID1L = [6, 16]     #  'Lang1_LH_netw' 'MD1_LH_netw'
        select_fID1R = [22, 32]     #  'Lang1_RH_netw' 'MD1_RH_netw' 
        select_fID2 = [40, 52]     #  'Lang2_netw' 'MD2_netw'   
        select_fID3 = [59, 68]     #  'Lang3_netw'  'MD3_netw'


        Significant_Lang = [2, 20, 37]  
        Significant_MD_1L   = [7, 12, 13, 14, 15]
        Significant_MD_1R   = [23, 25, 29, 30]
        Significant_MD_2    = [42, 43, 44, 46, 51]
        Significant_MD_3   = [60, 61, 62, 63, 68]

        Part_list = [Significant_Lang, Significant_MD_3]; app = '_Lang_MD3'
        # Part_list = [Significant_MD_1L, Significant_MD_1R]; app = '_MD1RL'
        # Part_list = [Significant_MD_2, Significant_MD_3]; app = '_MD23'

        fID_LangMD = {
            '1_LH': {'Lang': [1,6], 'MD': [7,16]},
            '1_RH': {'Lang': [17,22], 'MD': [23,32]},
            '2': {'Lang': [33, 40], 'MD': [41,52]},
            '3': {'Lang': [53, 59], 'MD': [60,68]},
        }

        return features, select_fID1L, select_fID1R, select_fID2, select_fID3, Part_list, app
    
    features, select_fID1L, select_fID1R, select_fID2, select_fID3, Part_list, app = analyze_fMRI_data()

    fID_LM2 = {'fID': select_fID2, 'label': 'LM2'}
    fID_LM3 = {'fID': select_fID3, 'label': 'LM3'}



    Dir_save  = '../../results/fMRI'

    # check_StandardScaler_for_fMRI()
    # record_Plt_ROI_fMRI_case(Dir_save)


    def check_all_plot():
        from matplotlib import pyplot as plt
        from Plt_compare import Plt_select_ROI_Wd_C01, select_2ROI_data

        # Plt_2ROI_compare(dataset.Avg['fMRI_C01'], 
        #         features, select_fID1L, select_fID1R, select_fID2, select_fID3, Dir_save, 'fMRI_avg', 'fMRI avg')

        fMRI_or_pred_data = dataset.Avg['fMRI_C01']
        title = 'fMRI avg'
        select_fID = select_fID3

        Plt_select_ROI_Wd_C01(dataset.Avg['fMRI_C01'], features, fID_LM3, Dir_save)
        
        # ROI_values = fMRI_or_pred_data['C0']
        # xy_data, x_data_label, y_data_label = \
        #         Get_two_columns_data(ROI_values, features, select_fID)
        # u_data, v_data = uv_Rotation(x_data_label, y_data_label)
    check_all_plot()

    # ----------------------------- #
    #   do statistics               #
    # ----------------------------- #

    def check_all_statistics():    
        from Plt_compare import Get_two_columns_data
        
        from calc_statistics import MV_T2_statistic, Print_Hotelling
        from calc_statistics import hotellings_t2_v0, hotellings_t2

        from calc_statistics import Plt_Boxplot, Plt_box_with_p, Plt_box_with_p_v2
        from calc_statistics import Cohens_d_barplot
        from calc_statistics import Plt_PCA_TSNE, Plt_PCA
        from calc_statistics import compare_plot_cov_ellipse
        from calc_statistics import groupAB2df
        from calc_statistics import find_Hotelling_statistics


        fMRI_or_pred_data = dataset.Avg['fMRI_C01']
        title = 'fMRI avg'
        select_fID = select_fID3

        group_C0, C0_x, C0_y = Get_two_columns_data(fMRI_or_pred_data['C0'], features, select_fID)
        group_C1, C1_x, C1_y = Get_two_columns_data(fMRI_or_pred_data['C1'], features, select_fID)

        only_two_features = 0
        if only_two_features == 1:
            group_A = group_C0; roi_A = C0_x[1]
            group_B = group_C1; roi_B = C0_y[1]

            Plt_pairplot(group_A, group_B)

            # ------ Hotelling's T2 test ------- #
            results_T2_F, res, result_MV_T2 = find_Hotelling_statistics(group_A, group_B)

        elif only_two_features == 0:

            group_A = fMRI_or_pred_data['C0']; roi_A = FMRI_key_list[59-1]; rLabel = 'Ln3nt'
            group_B = fMRI_or_pred_data['C1']; roi_B = FMRI_key_list[68-1]; rLabel = rLabel + '_MD3nt'

            results_ROI_all, ROI_all, AB_label, df, result_MV_T2 = Plt_box_with_p_v2(group_A, group_B, \
                # option=2, fID_LangMD=fID_LangMD, Dir_save=DirFig, Dtype='fMRI')
                option=3, Part_list=Part_list, Dir_save=Dir_save, append=app, Dtype='fMRI')

        # result_MV_T2_v2 = Cohens_d_barplot(group_A, group_B)   # plot all feartures' Cohen's d

        def Plt_two_features(roi_A, roi_B):
            # select two features to plot: roi_A, roi_B
            df, df_melt = Plt_Boxplot(group_A, group_B, roi_A, roi_B)
            results, features_PltAB = Plt_box_with_p(df, df_melt, result_MV_T2, Dir_save=Dir_save, rLabel=rLabel, Dtype='fMRI')   # feature of roi_A, roi_B

            return results, features_PltAB
        # results, features_PltAB = Plt_two_features(roi_A, roi_B)


        def Plt_PCA_to_check():
            # ------ PCA two dimension plot ------- #
            Plt_PCA_TSNE(group_A, group_B)
            Plt_PCA(group_A, group_B, result_MV_T2)
            compare_plot_cov_ellipse(group_A, group_B, result_MV_T2)
        # Plt_PCA_to_check()
    
    check_all_statistics()


# %%


