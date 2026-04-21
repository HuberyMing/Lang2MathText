
from matplotlib import pyplot as plt
import os
import pandas as pd

def Plt_straight_x_equal_y(axNow, Option=0):
    Lx = axNow.get_xlim()
    Ly = axNow.get_ylim()
    Lxy = Lx if max(Lx) > max(Ly) else Ly

    if Option == 0:
        axNow.plot(Lxy, Lxy, 'r--', alpha=0.5, label='y=x')
    elif Option == 1:

        # mLxy = [-x for x in Lxy]
        mLxy = (-Lxy[0], -Lxy[1])

        axNow.plot(mLxy, Lxy, 'r--', alpha=0.5, label='y= -x')

    axNow.legend(loc='upper left', shadow=False, fontsize=7)


def Plt_ROI(features, 
        df_xdata, xlabel,
        df_ydata, ylabel,
        axNow, 
        leg: str,
        select_ROI = 68,
        shift = 0.0,
        color = 'b',
        ):

    if not (df_xdata.index == df_ydata.index).all():    
        # df_xdata = df_xdata.sort_values(by='stimsetid')
        # df_ydata = df_ydata.sort_values(by='stimsetid')

        columns = df_xdata.columns
        df_xdata = df_xdata.sort_values(by=columns[1]).sort_values(by=columns[2])
        df_ydata = df_ydata.sort_values(by=columns[1]).sort_values(by=columns[2])

        assert (df_xdata.index == df_ydata.index).all()

    xdata = df_xdata.iloc[:, select_ROI-1]
    ydata = df_ydata.iloc[:, select_ROI-1]

    ydata = ydata + shift


    ROI_name = features[select_ROI]

    axNow.scatter(xdata, ydata, color=color, label=leg)

    axNow.set_xlabel(xlabel)
    axNow.set_ylabel(ylabel)
    axNow.set_title(ROI_name)

    axNow.legend(loc='upper left', shadow=False, fontsize=7)

    Plt_xy = 0
    if Plt_xy == 1:
        Plt_straight_x_equal_y(axNow)

def select_2ROI_data(fMRI, features, select_fID):

    xId, yId = select_fID

    print(f'select_fID = {select_fID}')
    print(f'    xId  = {xId}')
    print(f'    yId  = {yId}')


    xlabel = features[xId]
    ylabel = features[yId]

    assert xlabel == fMRI.columns[xId-1]
    assert ylabel == fMRI.columns[yId-1]

    xdata = fMRI.iloc[:, xId-1]
    ydata = fMRI.iloc[:, yId-1]

    return (xdata, xlabel), (ydata, ylabel)


def Get_two_columns_data(ROI_values, features, select_fID):

    x_data_label, y_data_label = select_2ROI_data(ROI_values, features, select_fID)
    
    (xdata, xlabel) = x_data_label
    (ydata, ylabel) = y_data_label
    xy_data = pd.concat((xdata, ydata), axis=1)

    return xy_data, x_data_label, y_data_label



def Plt_2ROI(fMRI, features, select_fID, axNow, leg=''):

    (xdata, xlabel), (ydata, ylabel) = select_2ROI_data(fMRI, features, select_fID)

    axNow.scatter(xdata, ydata, label=leg)
    
    axNow.set_xlabel(xlabel)
    axNow.set_ylabel(ylabel)

    axNow.legend(loc='upper left', shadow=False, fontsize=7)

    axNow.set_box_aspect(1)
    axNow.set_aspect('equal')
    axNow.grid(True, which='both')

    axNow.axhline(y=0, color='k')
    axNow.axvline(x=0, color='k')


def Plt_ROI_performance(fMRI_data, predict, features, select_MD, 
                labels = ['fMRI', 'predict'],
                Dir_save = './', Plt_case = 'ALL', ToShift=1):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.subplots_adjust(left=0.08, bottom=0.08, top=0.95, wspace=0.2, hspace=0.2)

    if 'data' in fMRI_data:
        fMRI = fMRI_data['data']
        lab_fMRI = fMRI_data['split'] + ' ' + fMRI_data['from']
    elif 'C0' in fMRI_data:
        fMRI = fMRI_data
        lab_fMRI = labels[0] #'fMRI'

    if 'data' in predict:
        Pred = predict['data']
        lab_pred = predict['split'] + ' ' + predict['from']
    elif 'C0' in predict:
        Pred = predict
        lab_pred = labels[1] #'predict'


    for ii, featID in enumerate(select_MD):
        ix = ii%2
        iy = int(ii/2)

        Plt_ROI(features, fMRI['C0'], lab_fMRI, Pred['C0'], lab_pred, ax[iy, ix], 'C0', featID, color='b')
        Plt_ROI(features, fMRI['C1'], lab_fMRI, Pred['C1'], lab_pred, ax[iy, ix], 'C1', featID, color='r')

        if ToShift == 1:
            Plt_ROI(features, fMRI['C0'], lab_fMRI, Pred['C0'], lab_pred, ax[iy, ix], 'C0', featID,-0.03, color='b')
            Plt_ROI(features, fMRI['C1'], lab_fMRI, Pred['C1'], lab_pred, ax[iy, ix], 'C1', featID, 0.03, color='r')

        Plt_straight_x_equal_y(ax[iy, ix])

    DirFig = f'{Dir_save}/ROI_performance'

    os.makedirs(DirFig, exist_ok=True)
    plt.savefig(f'{DirFig}/ROI_{Plt_case}.png')
    # plt.show()


def Plt_2ROI_compare(fMRI_or_pred_data, features, 
                select_fID1L, select_fID1R, select_fID2, select_fID3,
                Dir_save='./', Plt_case='ALL', title = ''):
    """ compare two features in select_fID...
        fro the same data fMRI_or_pred
    Args:
        fMRI_or_pred (pd.DataFrame): fMRI or prediction
                --> only the same single data in this plot
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))
    # fig, ax = plt.subplots()

    if 'data' in fMRI_or_pred_data:
        fMRI_or_pred = fMRI_or_pred_data['data']
        tit = fMRI_or_pred_data['split'] + ' ' + fMRI_or_pred_data['from']
    elif 'C0' in fMRI_or_pred_data:
        fMRI_or_pred = fMRI_or_pred_data
        tit = title     #'fMRI'



    Plt_2ROI(fMRI_or_pred['C0'], features, select_fID1L, ax[0,0], leg='C0')
    Plt_2ROI(fMRI_or_pred['C1'], features, select_fID1L, ax[0,0], leg='C1')

    Plt_2ROI(fMRI_or_pred['C0'], features, select_fID1R, ax[0,1], leg='C0')
    Plt_2ROI(fMRI_or_pred['C1'], features, select_fID1R, ax[0,1], leg='C1')

    Plt_2ROI(fMRI_or_pred['C0'], features, select_fID2, ax[1,0], leg='C0')
    Plt_2ROI(fMRI_or_pred['C1'], features, select_fID2, ax[1,0], leg='C1')

    Plt_2ROI(fMRI_or_pred['C0'], features, select_fID3, ax[1,1], leg='C0')
    Plt_2ROI(fMRI_or_pred['C1'], features, select_fID3, ax[1,1], leg='C1')

    ax[0,0].set_title(tit)

    Plt_straight_x_equal_y(ax[0, 0])
    Plt_straight_x_equal_y(ax[0, 1])
    Plt_straight_x_equal_y(ax[1, 0])
    Plt_straight_x_equal_y(ax[1, 1])

    Plt_straight_x_equal_y(ax[0, 0], Option=1)
    Plt_straight_x_equal_y(ax[0, 1], Option=1)
    Plt_straight_x_equal_y(ax[1, 0], Option=1)
    Plt_straight_x_equal_y(ax[1, 1], Option=1)


    DirFig = f'{Dir_save}/TwoROI'
    os.makedirs(DirFig, exist_ok=True)    
    plt.savefig(f'{DirFig}/Two_ROI_{Plt_case}.png')
    # plt.show()

# -------------------------------- #
#      for plot sections           #
# -------------------------------- #


def section_of_xyData(xdata, ydata):
    common_index = xdata.index.intersection(ydata.index)

    xpIdx = xdata[xdata>0].index
    xmIdx = xdata[xdata<0].index
    ypIdx = ydata[ydata>0].index
    ymIdx = ydata[ydata<0].index

    # assert (xpIdx.union(xmIdx) == common_index).all()
    # assert (ypIdx.union(ymIdx) == common_index).all()
    assert (sorted(xpIdx.union(xmIdx)) == sorted(common_index))
    assert (sorted(ypIdx.union(ymIdx)) == sorted(common_index))

    xGTyp = common_index[xdata > ydata]
    xLTyp = common_index[xdata < ydata]
    xGTym = common_index[xdata > -ydata]        #  x > -y  <==>    -x < y
    xLTym = common_index[xdata < -ydata]         #  x < -y  <==>    -x > y
    # assert (xGTyp.union(xLTyp) == common_index).all()
    # assert (xGTym.union(xLTym) == common_index).all()
    assert (sorted(xGTyp.union(xLTyp)) == sorted(common_index))
    assert (sorted(xGTym.union(xLTym)) == sorted(common_index))

    Quadrant = {
            1: {'Idx': xpIdx.intersection(ypIdx), 'label': 'Q1'},
            2: {'Idx': xmIdx.intersection(ypIdx), 'label': 'Q2'},
            3: {'Idx': xmIdx.intersection(ymIdx), 'label': 'Q3'},
            4: {'Idx': xpIdx.intersection(ymIdx), 'label': 'Q4'}
        }
    Direction = {
        'e': {'Idx': xGTyp.intersection(xGTym), 'label': 'e'},
        'w': {'Idx': xLTyp.intersection(xLTym), 'label': 'w'},
        's': {'Idx': xGTyp.intersection(xLTym), 'label': 's'},
        'n': {'Idx': xLTyp.intersection(xGTym), 'label': 'n'}
    }

    QD = {
        'E1': { 'Idx': Quadrant[1]['Idx'].intersection(Direction['e']['Idx']),
                'label': 'E1'
            },
        'N1': { 'Idx': Quadrant[1]['Idx'].intersection(Direction['n']['Idx']),
                'label': 'N1'
            },
        'N2': { 'Idx': Quadrant[2]['Idx'].intersection(Direction['n']['Idx']),
                'label': 'N2'
            },
        'W2': { 'Idx': Quadrant[2]['Idx'].intersection(Direction['w']['Idx']),
                'label': 'W2'
            },
        'W3': { 'Idx': Quadrant[3]['Idx'].intersection(Direction['w']['Idx']),
                'label': 'W3'
            },
        'S3': { 'Idx': Quadrant[3]['Idx'].intersection(Direction['s']['Idx']),
                'label': 'S3'
            },
        'S4': { 'Idx': Quadrant[4]['Idx'].intersection(Direction['s']['Idx']),
                'label': 'S4'
            },
        'E4': { 'Idx': Quadrant[4]['Idx'].intersection(Direction['e']['Idx']),
                'label': 'E4'
            }
    }
    
    return Quadrant, Direction, QD


def Plt_xy_section(axNow, x_data_label, y_data_label, select_Idx_label):

    (xdata, xlabel) = x_data_label
    (ydata, ylabel) = y_data_label

    select_Idx = select_Idx_label['Idx']
    label      = select_Idx_label['label']
    legend     = f'{label}: # {select_Idx.shape[0]}'

    axNow.scatter(xdata[select_Idx], ydata[select_Idx], label=legend)
    
    axNow.set_xlabel(xlabel)
    axNow.set_ylabel(ylabel)

    axNow.legend(loc='upper left', shadow=False, fontsize=7)

    axNow.set_box_aspect(1)
    axNow.set_aspect('equal')
    axNow.grid(True, which='both')

    axNow.axhline(y=0, color='k')
    axNow.axvline(x=0, color='k')

def Plt_select_Wd_C01(axNow, features, select_fID, ROI_values, tit, Option = 0):
    x_data_label, y_data_label = select_2ROI_data(ROI_values, features, select_fID)

    (xdata, xlabel) = x_data_label
    (ydata, ylabel) = y_data_label

    Quadrant, Direction, QD = section_of_xyData(xdata, ydata)

    if Option == 0:
        for item in Quadrant:
            Plt_xy_section(axNow, x_data_label, y_data_label, Quadrant[item])
        # Plt_xy_section(axNow, x_data_label, y_data_label, Quadrant[1])
        # Plt_xy_section(axNow, x_data_label, y_data_label, Quadrant[2])    
        # Plt_xy_section(axNow, x_data_label, y_data_label, Quadrant[3])
        # Plt_xy_section(axNow, x_data_label, y_data_label, Quadrant[4])

    elif Option == 1:
        for item in Direction:
            Plt_xy_section(axNow, x_data_label, y_data_label, Direction[item])

    elif Option == 2:
        Plt_xy_section(axNow, x_data_label, y_data_label, QD['E1'])

        Plt_xy_section(axNow, x_data_label, y_data_label, QD['N1'])
        Plt_xy_section(axNow, x_data_label, y_data_label, QD['N2'])

        Plt_xy_section(axNow, x_data_label, y_data_label, QD['W2'])
        Plt_xy_section(axNow, x_data_label, y_data_label, QD['W3'])

        Plt_xy_section(axNow, x_data_label, y_data_label, QD['S3'])
        Plt_xy_section(axNow, x_data_label, y_data_label, QD['S4'])

        Plt_xy_section(axNow, x_data_label, y_data_label, QD['E4'])

    axNow.set_title(tit)

    Plt_straight_x_equal_y(axNow)
    Plt_straight_x_equal_y(axNow, Option=1)



def Plt_select_ROI_Wd_C01(fMRI_or_pred_data, features, fID_LM, Dir_save, Plt_case='MRIorPred', ):
    """
    (eg)
        fID_LM = fID_LM3 := {'fID': select_fID3, 'label': 'LM3'}

    """
    select_fID = fID_LM['fID']
    label      = fID_LM['label']

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 9))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.96, wspace=0.28, hspace=0.28)

    Plt_select_Wd_C01(ax[0,0], features, select_fID, fMRI_or_pred_data['C0'], 'C0')
    Plt_select_Wd_C01(ax[0,1], features, select_fID, fMRI_or_pred_data['C1'], 'C1')

    Plt_select_Wd_C01(ax[1,0], features, select_fID, fMRI_or_pred_data['C0'], 'C0', Option=1)
    Plt_select_Wd_C01(ax[1,1], features, select_fID, fMRI_or_pred_data['C1'], 'C1', Option=1)

    Plt_select_Wd_C01(ax[2,0], features, select_fID, fMRI_or_pred_data['C0'], 'C0', Option=2)
    Plt_select_Wd_C01(ax[2,1], features, select_fID, fMRI_or_pred_data['C1'], 'C1', Option=2)

    DirFig = f'{Dir_save}/TwoROI'
    os.makedirs(DirFig, exist_ok=True)    

    FigSave = f'{DirFig}/Two_ROI_{Plt_case}_sec{label}.png'
    print(f'FigSave = {FigSave}')

    plt.savefig(FigSave)
