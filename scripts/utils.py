import os
import copy
import time
import numpy as np
import pandas as pd
from collections import Counter 
import random
import natsort
import imageio
import imageio.v3 as iio
import tifffile
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata, Rbf

from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# note bool(np.nan)=True


# files io function
def read_file(file='.npy',para=None):
    if file.endswith('.csv'):
        return pd.read_csv(file,index_col=0,header=0) 
    elif file.endswith('.npy'):
        return np.load(file)
    elif file.endswith('.png'):
        data = imageio.imread(file)
        if data.shape[-1]==3 and para==4:
            new_shape = list(data.shape[:-1]) + [1]
            new_data = np.concatenate((data,255*np.ones(new_shape)),axis=-1)
            return new_data
        else:
            return data
    elif file.endswith('.tif') or file.endswith('.tiff'):
        data = tifffile.imread(file)
        if para==1:
            data = data.astype(float)
            data = normalize(data,by='total',method='std')
        return data
    elif file.endswith('.mat'):
        data = loadmat(file)['data'] 
        return data
    else:
        raise

def save_data(file,data=[],para=None,ifprint=True):
    os.makedirs(os.path.dirname(file),exist_ok=True)
    if file.endswith('.tif') and type(data)==np.ndarray:
        tifffile.imsave(file,data)
    elif file.endswith('.npy') and type(data)==np.ndarray:
        np.save(file,data)
    elif file.endswith('.csv') and type(data)==pd.core.frame.DataFrame:
        data.to_csv(file)
    elif file.endswith('.png') and len(data)==0:
        plt.savefig(file,dpi=400)
        plt.close()
    elif file.endswith('.png') and type(data)==np.ndarray:
        imageio.imwrite(file,data)
    elif file.endswith('.mp4') and type(data)==np.ndarray:
        iio.imwrite(file,data,fps=para,)
    else:
        raise
    if ifprint:
        print(rf'saved to {file}')



# data process function
def lim_data(data,lim=[0,1]):
    # 1D/2D data, simple limitation to restrict extender to vmax or vmin, lim length=2
    vmin,vmax = lim
    data[data>=vmax] = vmax
    data[data<=vmin] = vmin
    return data
def extract_triu(data,s=None,k=1,return_type='value',extract_type='median'):
    # data should be 2D, s is to restrict the data range of interest, only useful when type(s)==list
    # k should be ignored number, return_type could be 'value' or 'list'
    # note this is only for upper triangle
    if type(data)==pd.core.frame.DataFrame:
        data = data.values
    if type(data)==np.ndarray:
        if len(data.shape)==2:
            if data.shape[0]==data.shape[1]:
                if s!=None: 
                    m = np.meshgrid(s,s)
                    data = data[m[0],m[1]]
                index = np.ones(data.shape)
                index = np.triu(index,k=k).astype(bool)
                dList = data[index]
                if return_type=='list': return dList
                if return_type=='value': 
                    if extract_type=='median': return np.nanmedian(dList)
                    elif extract_type=='mean': return np.nanmean(dList)
                    elif extract_type=='max': return np.nanmax(dList)
                    else: raise
    raise
def flatten_data(data):
    # input should only be numpy or dataframe or list of them
    # row1 first, row2 second ..., return as np.adarray
    if type(data)==np.ndarray:
        newData = data.flatten()
    elif type(data)==pd.core.frame.DataFrame:
        # columns = []
        # for i in data.index:
        #     for c in data.columns:
        #         columns.append(f'{i}_{c}')
        # data = data.values.flatten()
        # data = pd.DataFrame(np.array([data]),columns=columns)
        newData = data.values.flatten()
    elif type(data)==list:
        # note it is different
        newData = []
        for d in data:
            newData.append(flatten_data(d))
    else:
        raise
    return newData 
def sort(data,by):
    if type(data)==dict and by=='value':
        # note, value is number, and values are all different from each other
        keys = np.array(list(data.keys()))
        values = np.array(list(data.values()))
        sortValues = natsort.natsorted(values)
        newData = {}
        for v in sortValues: 
            ks = keys[values==v]
            for k in ks:
                if k not in newData.keys(): 
                    newData[k] = v
                    break
        return newData
    else:
        raise
def argsort_data(data,inverse=False):
    # input can be dataframe or any dimension ndarray
    # output is tuple of index ndarray
    if type(data)==np.ndarray:
        new_data = data.copy()
    elif type(data)==pd.core.frame.DataFrame:
        new_data = data.values
    elif type(data)==list:
        new_data = np.array(data)
    else:
        raise
    shape = data.shape
    index = np.argsort(new_data,axis=None) # np.nan will be put in the last
    if inverse: index = index[::-1] # np.nan will be put in the first
    multi_dim_index = np.unravel_index(index, shape) 
    return multi_dim_index
def numNAN(data):
    # input 2d dataframe or any dimension ndarray or singe number
    # return number only
    try:
        len(data) # single number has no length
        if type(data)==np.ndarray:
            return np.sum(np.isnan(data))
        elif type(data)==pd.core.frame.DataFrame:
            return np.sum(np.isnan(data.values))
        else:
            raise
    except:
        return 1 if np.isnan(data) else 0
def isnan(data):
    # input 2d dataframe or any dimension ndarray or singe number
    # return True/False only
    try:
        len(data) # single number has no length
        if type(data)==np.ndarray:
            return np.sum(np.isnan(data))==np.prod(data.shape)
        elif type(data)==pd.core.frame.DataFrame:
            return np.sum(np.isnan(data.values))==np.prod(data.shape)
        else:
            raise
    except:
        return np.isnan(data)
def replace_data(data,para0,para1):
    # input ndarray or dataframe, output is the same type
    if np.isnan(para0):
        if type(data)==np.ndarray:
            new_data = np.nan_to_num(data,nan=para1)
        elif type(data)==pd.core.frame.DataFrame:
            new_data = data.fillna(para1)
        else:
            raise
    else:
        if type(data) in [np.ndarray,pd.core.frame.DataFrame]:
            new_data = data.copy()
            new_data[data==para0] = para1
        else:
            raise
    return new_data
def fillna_data_23D(data,axis=-1):
    # 2D array or 2D dataframe or 3D array
    dfFlag = False
    if type(data)==pd.core.frame.DataFrame:
        dfFlag = True
        index = data.index
        columns = data.columns
        data = data.values
    naPosition = np.unique(np.argwhere(np.isnan(data))[:,axis])
    if len(naPosition):
        if len(data.shape)==2 and axis==0:
            data[naPosition,:] = np.nanmedian(data,axis=axis)
        if len(data.shape)==2 and axis==1:
            data[:,naPosition] = np.nanmedian(data,axis=axis)
        if len(data.shape)==3 and axis==0:
            data[naPosition,:,:] = np.nanmedian(data,axis=axis)
        if len(data.shape)==3 and axis==1:
            data[:,naPosition,:] = np.nanmedian(data,axis=axis)
        if len(data.shape)==3 and axis==2:
            data[:,:,naPosition] = np.nanmedian(data,axis=axis)
    if dfFlag:
        data = pd.DataFrame(data,index=index,columns=columns)
    return data
def cat_data_23D(data,axises=[-1,-1],restrictions=[[],[]]):
    # 2D array or 2D dataframe or 3D array
    dfFlag = False
    if type(data)==pd.core.frame.DataFrame:
        dfFlag = True
        index = data.index
        columns = data.columns
        data = data.values
    for axis,restriction in zip(axises,restrictions):
        if len(data.shape)==2 and axis==0:
            data = data[restriction,:]
        if len(data.shape)==2 and axis==1:
            data = data[:,restriction]
        if len(data.shape)==3 and axis==0:
            data = data[restriction,:,:]
        if len(data.shape)==3 and axis==1:
            data = data[:,restriction,:]
        if len(data.shape)==3 and axis==2:
            data = data[:,:,restriction]
    if dfFlag:
        for axis,restriction in zip(axises,restrictions):
            if axis==0: index = index[restriction]
            if axis==1: columns = columns[restriction]
        return pd.DataFrame(data,index=index,columns=columns)
    else:
        return data
def median_data(input,axis=-1):
    # support dataframe and ndarray, make sure every data is same size
    if type(input[0])==pd.core.frame.DataFrame:
        input_ = [i.values for i in input]
        output_ = np.nanmedian(np.array(input),axis=0)
        output = pd.DataFrame(output_,index=input[0].index,columns=input[0].columns)
    elif type(input[0])==np.ndarray:
        output = np.nanmedian(np.array(input),axis=0)
    else:
        raise
    return output
def normalize(input=None,
              by='col',
              method='sum'):
    '''
    Parameter
    ---input: dataframe or array of 2-d
    ---by: normalization by row or col or total
    ---method: normalized to sum(bys)=1 or std(by_vmin=0,by_vmax=1) or just max(by)=1
    Return:
    ---the same shape and type of the input
    Notes:
    ---If 1d/3d input, only by=total is supported
    ---the /0=np.nan problem to be solved
    '''
    dfFlag = False
    if type(input)==pd.core.frame.DataFrame:
        dfFlag = True
        index = input.index
        columns = input.columns
        input = input.values
    
    if method=='sum':
        
        if by=='col':  
            bySum = np.nansum(input,axis=1).T
            byDivide = np.tile(bySum,(input.shape[1],1)).T 
            output = input / byDivide
        elif by=='row':
            output = input / np.nansum(input,axis=0)
        elif by=='total':
            sum_ = np.nansum(input)
            output = input/sum_
    
    elif method=='std':
        if by=='col':  
            output = ( input - np.nanmin(input,axis=0) ) / ( np.nanmax(input,axis=0) - np.nanmin(input,axis=0) )
        elif by=='row':
            input_ = input.T
            output_ = ( input_ - np.nanmin(input_,axis=0) ) / ( np.nanmax(input_,axis=0) - np.nanmin(input_,axis=0) )
            output = output_.T        
        elif by=='total':
            max_ = np.nanmax(input)
            min_ = np.nanmin(input)
            output = (input - min_) / (max_ - min_)
    
    elif method=='max':
        if by=='col':  
            output = input / np.nanmax(abs(input),axis=0)
        elif by=='row':
            input_ = input.T
            output_ = input_ / np.nanmax(abs(input_),axis=0)
            output = output_.T        
        elif by=='total':
            max_ = np.nanmax(abs(input))
            output = input/max_

    if dfFlag:
        output = pd.DataFrame(output,index=index,columns=columns)
    if np.nansum(np.isnan(input))!=np.nansum(np.isnan(output)): 
        print('note divide 0 here and there are nan values in output')
        # raise 
    return output
def smooth(data, para,
           method='rolling'):
    # data is 1d numpy.ndarray without np.nan
    # para is different in different method
    # method can be 'rolling' or 'weight' or 'exp'
    if np.nansum(np.isnan(data)) > 0: raise
    if method=='rolling': # newData = smooth(data,para=5,method='rolling')
        window_size = para
        series = pd.Series(data)
        newData = series.rolling(window=window_size).mean()
    elif method=='weight': # newData = smooth(data,para=[0.1,0.2,0.4,0.2,0.1],method='weight')
        weights = para
        series = pd.Series(data)
        newData = series.rolling(window=len(weights)).apply(lambda x: np.sum(x * weights))
    elif method=='exp': # newData = smooth(data,para=0.2,method='exp')
        alpha = para
        model = SimpleExpSmoothing(data)
        result = model.fit(smoothing_level=alpha, optimized=False)
        newData = result.fittedvalues
    elif method=='savgol': # newData = smooth(data,para=[3,1],method='savgol')
        window_size, polyorder = para
        newData = savgol_filter(data, window_size, polyorder)
    else:
        raise
    return newData
def find_center(Data):
    shapes = Data.shape
    dim = len(shapes)
    center = []
    for centerAxis in range(dim):
        data = Data.copy()
        for i in range(centerAxis-1,-1,-1):
            data = np.nansum(data,axis=0)
        for i in range(centerAxis+1,dim,1):
            data = np.nansum(data,axis=1)
        centerAxisValue = np.nanargmax(data)
        center.append(centerAxisValue)
    return center
def generate_kernel(kshape,ktype='sum'):
    # generate a 3d kernal with max value=1
    shape1,shape2,shape3 = kshape
    if type(shape1) != int or type(shape2) != int or type(shape3) != int: raise
    if shape1%2 != 1 or shape2%2 != 1 or shape3%2 != 1: raise
    if ktype=='sum':
        kernel = np.ones((shape1,shape2,shape3))
    elif ktype=='center':
        kernel = np.zeros((shape1,shape2,shape3))
        center1,center2,center3 = ((np.array([shape1,shape2,shape3])+1)/2).astype(int)
        for i in range(shape1):
            for j in range(shape2):
                for k in range(shape3):
                    v = max(abs(center1-(i+1)),abs(center2-(j+1)),abs(center3-(k+1)))
                    kernel[i,j,k] = 1/(2**v)
    else:
        raise
    return kernel
def draw_projectionNODE(ax=None,
                        figfile='',
                        boundarysize=0.24,boundaryshifts=[-0.005,-0.0005],boundarypointsize=1,
                        locs=[],types=[],projs=[],weights=[],
                        scattersizebase=1,
                        linetype='',          
                        linewidth=0.1,                        
                        projlimit=None,cmap=None,nc=10,alpha=1,
                        ):
    if bool(figfile):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1,1,1)
    # boundary plot
    boundary = imageio.imread(r'H:\BCIteam_Allrelated\SharedSource\bciBASE\boundary.png')
    indicies = np.nonzero(boundary[:,:,0]<120)
    scale = boundarysize/boundary.shape[0]
    sns.scatterplot(x=scale*indicies[1]+boundaryshifts[0],y=scale*(boundary.shape[0]-indicies[0])+boundaryshifts[1],c='black',s=boundarypointsize,ax=ax)
    # prepare
    tprojs = {t:np.nanmedian(projs[types==t]) for t in set(types)}
    tprojs = sort(tprojs,by='value')
    nc = len(tprojs)
    newProjs = [tprojs[t] for t in types]
    if projlimit==None: projlimit = [np.nanmin(newProjs),np.nanmax(newProjs)]
    indexTcenters = {t:np.nanargmax(weights[types==t]) for t in tprojs.keys()}
    tcenters = {t:(locs[types==t])[indexTcenters[t]] for t in tprojs.keys()}
    # plot
    typeDrawList = list(tprojs.keys())
    for it in range(nc):
        t = typeDrawList[it]
        tproj = tprojs[t]
        color = get_colorcode(value=tproj,value_range=projlimit,cmap=cmap,nc=nc)
        tlocs = locs[types==t]
        sns.scatterplot(x=tlocs[:,0],y=tlocs[:,1],c=color,s=scattersizebase,alpha=alpha,ax=ax)     
        if it!=(nc-1):
            tcenter = tcenters[t]
            nextt = typeDrawList[it+1]
            nexttcenter = tcenters[nextt]
            if linetype=='astral':
                ax.plot([tcenter[0],nexttcenter[0]],
                        [tcenter[1],nexttcenter[1]],
                        color='black',
                        linewidth=linewidth,
                        alpha=alpha)
            elif linetype=='arrow':
                ax.arrow(tcenter[0],tcenter[1],
                        nexttcenter[0]-tcenter[0],nexttcenter[1]-tcenter[1],
                        color='black',
                        width=linewidth,
                        head_width=linewidth*10,
                        length_includes_head=False,
                        alpha=alpha)
            else:
                raise        
    ax.axis('off')
    if bool(figfile):
        save_data(figfile)
    return
def draw_projectionTOPO(
                    ax=None,
                    figfile='',
                    boundarysize=0.24,
                    boundaryshifts=[-0.005,-0.0005],
                    boundarypointsize=1,
                    start=None, # a row of montage dataframe
                    projs=None, # dataframe of montage
                    palette=[],
                    cmap=None,
                    nc=-1,
                    alpha=1,
                    plot='astral',
                    linewidth=0.1,
                    ):
    if bool(figfile):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1,1,1)
    #boundary plot
    boundary = imageio.imread(r'H:\BCIteam_Allrelated\SharedSource\bciBASE\boundary.png')
    indicies = np.nonzero(boundary[:,:,0]<120)
    scale = boundarysize/boundary.shape[0]
    sns.scatterplot(x=scale*indicies[1]+boundaryshifts[0],y=scale*(boundary.shape[0]-indicies[0])+boundaryshifts[1],c='black',s=boundarypointsize,ax=ax)
    #scatter
    sns.scatterplot(x=np.array(projs)[:,0],y=np.array(projs)[:,1],c='black',s=boundarypointsize,ax=ax)
    #line
    if np.nanmin(projs['value'])<0: 
        maxValue = np.nanmax(np.abs(projs['value']))
        minValue = -maxValue
        value_range = [-maxValue,maxValue]
    else:
        minValue = np.nanmin(projs['value'])
        maxValue = np.nanmax(projs['value'])
        value_range = [minValue,maxValue]
    for i,end in projs.iterrows():
        if len(palette)!=0:
            color = get_colorcode(end['value'],value_range,palette=palette,reverse=False) # # do not need nc
        elif cmap!=None:
            color = get_colorcode(end['value'],value_range=value_range,cmap=cmap,nc=nc)
        else:
            raise
        if plot=='astral':
            ax.plot([start['RIGHT'],end['RIGHT']],
                    [start['NOSE'],end['NOSE']],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha)
        elif plot=='arrow':
            ax.arrow(start['RIGHT'],start['NOSE'],
                    end['RIGHT']-start['RIGHT'],end['NOSE']-start['NOSE'],
                    color=color,
                    width=linewidth,
                    head_width=linewidth*10,
                    length_includes_head=False,
                    alpha=alpha)
        else:
             raise
    ax.axis('off')
    if bool(figfile):
        save_data(figfile)
    return
def rgb2hex(rgb):
    return '#'+"".join([hex(c)[2:].rjust(2,'0').upper() for c in rgb])
def get_colorcode(value,value_range,cmap='vlag',nc=10,palette=[],reverse=False): # value,value_range
    if len(palette)==0:
        rgblist = (np.array(sns.color_palette(cmap,nc+1))*255).astype(int)
    else:
        rgblist = palette
        nc = len(palette)-1
    if reverse:
        rgblist = rgblist[::-1,:]
    code16list = [rgb2hex(rgb) for rgb in rgblist]
    standardvalue = int(nc * (value-value_range[0])/(value_range[1]-value_range[0]))
    if standardvalue<0: standardvalue=0
    if standardvalue>nc: standardvalue=nc
    colorcode = code16list[standardvalue]
    return colorcode
# corr generating function
def calc_selfCorr(data,axis=1,method='pearson'):
    # only for 2D array or dataframe, axis is to adjust to row corr need, method can be 'pearson' or 'spearman'
    # note 'pearson' is more used, only 'spearsonman' when
    if type(data)==np.ndarray:
        if axis==0:
            data = data.T
        df_data = pd.DataFrame(data)
        corr_data = df_data.corr(method=method)
        return corr_data.values
    if type(data)==pd.core.frame.DataFrame:
        if axis==0:
            data = pd.DataFrame(data.values.T,index=data.columns,columns=data.index)
        return data.corr(method=method) 
def calc_pairCorr(rowData,colData,axis=1,method='pearson'):
    # only for 2D array or dataframe, axis is to adjust to row corr need, method can be 'pearson' or 'spearman'
    # note 'pearson' is more used, only 'spearsonman' when
    if not type(rowData)==type(colData): raise # note rowData and colData should be same type
    npyFlag = False
    if type(rowData)==np.ndarray:
        npyFlag = True
        rowData = pd.DataFrame(rowData)
        colData = pd.DataFrame(colData)
    if axis==0:
        rowData = pd.DataFrame(rowData.values.T,index=rowData.columns,columns=rowData.index)
        colData = pd.DataFrame(colData.values.T,index=colData.columns,columns=colData.index)
    columnsFlag = False
    if bool(set(rowData.columns)&set(colData.columns)):
        columnsFlag = True
        rowData.columns = [f'{i}_row' for i in rowData.columns]
        colData.columns = [f'{i}_col' for i in colData.columns]
    totalData = pd.DataFrame(np.concatenate([rowData.values,colData.values],axis=1))
    totalData.columns = list(rowData.columns) + list(colData.columns)
    totalData = normalize(totalData,by='col',method='max')
    corr = totalData.corr(method=method)
    corr = corr.drop(labels=rowData.columns,axis=1)
    corr = corr.drop(labels=colData.columns,axis=0)
    if columnsFlag:
        corr.index = [i[:-4] for i in corr.index]
        corr.columns = [i[:-4] for i in corr.columns]
    if npyFlag:
        return corr.values
    return corr



# files selection function
def select_files_byLabels(allFiles,allLabels,restrictions=[],labelRemain=True):
    files = copy.deepcopy(allFiles)
    labels = copy.deepcopy(allLabels)
    files = [f for f,l in zip(files,labels) if l in restrictions]
    if labelRemain:
        labels = [l for l in labels if l in restrictions]
    else:
        labels = [restrictions.index(l) for l in labels if l in restrictions]
    return files,labels



# montage related function
def get_locs_fromName(ch_names,return_type=''):
    # return ch_num*3(x,y,z)
    montage = read_file(rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\montage.csv')
    locs = montage.loc[ch_names]
    if return_type==3:
        return locs.values
    elif return_type==2:
        return locs.values[:,:2]
    else:
        return locs
def generate_montage_fromNumpy(locs):
    if type(locs)==list:
        locs = np.array(locs)
    # loc_type can be 2d or 3d
    loc_type = locs.shape[1]
    if not loc_type in [2,3]: raise
    df = pd.DataFrame([],columns=['RIGHT','NOSE','TOP'])
    if loc_type==2:
        locx, locy = locs.T
        locz = np.zeros(locx.shape)
    else:
        locx, locy, locz = locs.T
    i = 0
    for ix,iy,iz in zip(locx,locy,locz):
        df.loc[str(i)] = [ix,iy,iz]
        i += 1
    return df
def generate_montage_fromMesh(locs):
    # loc_type can be 2d or 3d
    loc_type = len(locs)
    if not loc_type in [2,3]: raise
    df = pd.DataFrame([],columns=['RIGHT','NOSE','TOP'])
    if loc_type==2:
        locx, locy = locs
        locz = np.zeros(locx.shape)
    else:
        locx, locy, locz = locs
    locx = flatten_data(locx)
    locy = flatten_data(locy)
    locz = flatten_data(locz)
    i = 0
    for ix,iy,iz in zip(locx,locy,locz):
        df.loc[str(i)] = [ix,iy,iz]
        i += 1
    return df
def countAxis(data,axis=None):
    # input should be ndarray
    # not used yet
    if not type(data)==np.ndarray: raise
    shapes = data.shape
    dim = len(shapes)
    axises = list(range(dim))
    axisWeights = []
    for iaxis,length in zip(axises,shapes):
        if (axis!=None and axis!=iaxis): continue
        axisWeight = np.arange(0,length,1)
        for i in range(iaxis-1,-1,-1):
            l = shapes[i]
            axisWeight = np.expand_dims(axisWeight, axis=0)
            axisWeight = np.repeat(axisWeight, repeats=l, axis=0)
        for i in range(iaxis+1,dim,1):
            l = shapes[i]
            axisWeight = np.expand_dims(axisWeight, axis=-1)
            weigthDim = len(axisWeight.shape)
            axisWeight = np.repeat(axisWeight, repeats=l, axis=-1)
        axisWeights.append(axisWeight)
        if (axis!=None and axis==iaxis): return axisWeight
    return axisWeights
def generate_mesh(step,ranges,inverse=False):
    # note the mesh is not same shape with data but filp xy
    # example: range [[0,3],[0,2],[0,5]] will generate [2,3,5]
    locSeries = []
    if inverse: ranges = ranges[::-1]
    for minl,maxl in ranges:
        minl = np.floor(minl/step)*step
        maxl = np.ceil(maxl/step)*step
        num = 1+(maxl-minl)/step
        seria = np.linspace(minl,maxl,int(num))
        locSeries.append(seria)
    locMesh = np.meshgrid(*locSeries)
    if inverse: locMesh = locMesh[::-1]
    return locMesh
def interpolate2D(locs,values,
                step=0.01,volumeLim=None,
                method='linear',fillna=np.nan):
    # locs are 3D x,y,z, values are same number with loc number
    # step is the new grid
    # method can be 'linear', 'nearest', 'cubic', 'rbf', 'gausian'
    minlxy = np.nanmin(locs,axis=0)
    maxlxy = np.nanmax(locs,axis=0)
    rangexy = [[i,j] for i,j in zip(minlxy,maxlxy)]
    locXy = generate_mesh(step,rangexy,inverse=False) # topo shape lengthx,lengthy, but mesh shape will be lengthy outer, lengthx inner, thus lenthy*lengthx
    locx,locy = locXy
    # note all ouput are lengthy*lengthx
    if method=='rbf':
        rbf = Rbf(locs[:,0],locs[:,1],values)
        interpolateValues = rbf(locx,locy)
    elif method=='gaussian':
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(locs, values)
        interpolateValues = gp.predict(np.c_[locx.ravel(),locy.ravel()]).reshape(locx.shape)
    elif method in ['linear','nearest', 'cubic']:
        interpolateValues = griddata(locs,values,(locx,locy),method=method)
    else:
        raise
    if volumeLim==None:
        return locXy,interpolateValues
    else:
        volumeXy = generate_mesh(step,volumeLim,inverse=False)
        volumex,volumey = volumeXy
        volumed_values = fillna*np.ones(volumeXy[0].shape)
        mx1,mx2 = locx[0,0],locx[-1,-1]
        vx1,vx2 = volumex[0,0],volumex[-1,-1]
        if vx1>mx1 or vx2<mx2: raise
        sx = int((mx1-vx1)/step)
        lx = int((mx2-mx1)/step)
        my1,my2 = locy[0,0],locy[-1,-1]
        vy1,vy2 = volumey[0,0],volumey[-1,-1]
        if vy1>my1 or vy2<my2: raise
        sy = int((my1-vy1)/step)
        ly = int((my2-my1)/step)
        volumed_values[sy:sy+ly+1,sx:sx+lx+1] = interpolateValues
        return volumeXy,volumed_values



# significance function
def confidence_interval(data):
    ci = stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    return ci
def significance_check(
                    data,
                    method='',
                    alpha=0.05,
                    para=None,
                    ):
    if method=='pearson': # not known what p_value is for yet
        # x,y are both list of values of same length
        x,y = data
        while 1 in np.array(x).shape: 
            x = flatten_data(x)
        while 1 in np.array(y).shape: 
            y = flatten_data(y)
        pearson, p_value = pearsonr(x,y)
        significance = p_value < alpha
        return pearson,p_value,significance
    if method=='rSquare': # only score, 1 is the best, while p_value<0.05 and 0 is the best
        # x,y are both list of values of same length
        x,y = data
        x = np.array(x).reshape(-1,1)
        while 1 in np.array(y).shape: 
            y = flatten_data(y)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        rSquare = r2_score(y, y_pred)
        return rSquare
    if method=='chiSquare': # data is an observed graph of row of one variable and a col of a variable
        # data is 2d, can also be shielded
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)
        significance = p_value < alpha
        return chi2_stat,p_value,significance
    if method=='zTest': # para is the population mean value to be the benchmark of 1-d data
        # data is a list, para is the value to be independent from
        while 1 in np.array(data).shape: 
            data = flatten_data(data)
        z_stat, p_value = ztest(data, value=para) 
        significance = p_value < alpha
        return z_stat,p_value,significance
    if method=='INDtTest': # for two data which has no link between
        # data1 and data2 is list
        data1,data2 = data
        while 1 in np.array(data1).shape: 
            data1 = flatten_data(data1)
        while 1 in np.array(data2).shape: 
            data2 = flatten_data(data2)
        t_stat,p_value = stats.ttest_ind(data1, data2)
        significance = p_value < alpha
        return t_stat,p_value,significance
    if method=='RELtTest':  # for two data which has link between
        # data1 and data2 is list of same length
        data1,data2 = data
        while 1 in np.array(data1).shape: 
            data1 = flatten_data(data1)
        while 1 in np.array(data2).shape: 
            data2 = flatten_data(data2)
        t_stat, p_value = stats.ttest_rel(data1, data2)
        significance = p_value < alpha
        return t_stat,p_value,significance
    if method=='onewayANOVA': # anova that out put 1
        # data is list of data where every list has different label, data must be 1d
        newData = []
        for data1 in data:
           while 1 in np.array(data1).shape: 
            data1 = flatten_data(data1)
            newData.append(data1)
        f_stat, p_value = stats.f_oneway(*newData)
        significance = p_value < alpha
        return f_stat,p_value,significance
    if method=='multiANOVA': # anova that output multiply
        # data is a dataframe with feature and labels columns, note test
        olsStr = f'{data.columns[-1]} ~ C({data.columns[0]})'
        for i in data.columns[1:-1]:
            olsStr += f' * C({i})'
        model = ols(olsStr, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=para) # para is the way to calc sum_sq with poible 1/2/3, should be carefilly coonsidered
        f_stat = anova_table['F']
        p_value = anova_table['PR(>F)']
        significance = p_value < alpha
        return f_stat,p_value,significance
    if method=='KruskalWallis': # not known yet
        # data is a dataframe with a 'feature' and a 'label' column, not test
        olsStr = f'feature ~ C(label)'
        model = ols(olsStr, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=para) # para is the way to calc sum_sq with poible 1/2/3, should be carefilly coonsidered
        groups = [data['feature'][data['label'] == label] for label in data['label'].unique()]
        kruskal_result = stats.kruskal(*groups)
        f_stat, p_value = kruskal_result
        significance = p_value < alpha
        return f_stat,p_value,significance



# seperabbility
def separability_measure(X1, X2, method='fisher'):
    # not used yet, need apply and correction on real data format
    """
    计算两组数据的可分性。
    
    参数:
        X1 (np.array): 第一组数据，形状为 (n_samples1, n_features)
        X2 (np.array): 第二组数据，形状为 (n_samples2, n_features)
        method (str): 使用的方法，可以是 'fisher', 'euclidean', 'mahalanobis', 'lda', 'kl_divergence', 'js_divergence', 'svm_margin', 'auc_svm'
        
    返回:
        float: 可分性度量

    注意：
        对于KL散度和JS散度 数据是一维的 如果数据是多维的 要对每个维度分别计算或者使用更复杂的密度估计方法
        对于SVM间隔 是一个理论上的概念 使用了支持向量机的最大间隔作为近似
        
    举例：
        X1 = np.random.randn(100, 2)  # 假设这是一个二维的数据集
        X2 = np.random.randn(100, 2) + np.array([3, 3])  # 第二组数据相对于第一组有位移
    """
    if method == 'fisher':
        # Fisher判别比
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        sb = (mean1 - mean2) @ (mean1 - mean2).T
        sw = np.cov(X1, rowvar=False) + np.cov(X2, rowvar=False)
        separability = sb / sw
        return separability.diagonal().sum()
    elif method == 'euclidean':
        # 欧几里得距离
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        separability = np.linalg.norm(mean1 - mean2)
        return separability
    elif method == 'mahalanobis':
        # 马氏距离
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        cov = np.cov(np.vstack((X1, X2)).T)
        inv_covmat = np.linalg.inv(cov)
        diff = mean1 - mean2
        separability = np.sqrt(diff @ inv_covmat @ diff.T)
        return separability
    elif method == 'lda':
        # 线性判别分析
        labels = np.concatenate([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])
        data = np.vstack((X1, X2))
        lda = LinearDiscriminantAnalysis()
        lda.fit(data, labels)
        separability = lda.coef_.ravel() @ (mean1 - mean2)
        return separability
    elif method == 'kl_divergence':
        # Kullback-Leibler散度
        density1 = gaussian_kde(X1.T)
        density2 = gaussian_kde(X2.T)
        x = np.linspace(X1.min(), X1.max(), 1000)
        kl_div = entropy(density1(x), density2(x))
        return kl_div
    elif method == 'js_divergence':
        # Jensen-Shannon散度
        density1 = gaussian_kde(X1.T)
        density2 = gaussian_kde(X2.T)
        x = np.linspace(X1.min(), X1.max(), 1000)
        m = 0.5 * (density1(x) + density2(x))
        js_div = 0.5 * (entropy(density1(x), m) + entropy(density2(x), m))
        return js_div
    elif method == 'svm_margin':
        # SVM间隔
        labels = np.concatenate([-np.ones(X1.shape[0]), np.ones(X2.shape[0])])
        data = np.vstack((X1, X2))
        svm = SVC(kernel='linear')
        svm.fit(data, labels)
        w = svm.coef_[0]
        b = svm.intercept_[0]
        # 平均间隔
        separability = 2 / np.linalg.norm(w)
        return separability
    elif method == 'auc_svm':
        # 使用SVM的预测概率来计算AUC
        labels = np.concatenate([-np.ones(X1.shape[0]), np.ones(X2.shape[0])])
        data = np.vstack((X1, X2))
        svm = SVC(kernel='linear', probability=True)
        svm.fit(data, labels)
        probas_ = svm.predict_proba(data)[:, 1]
        auc = roc_auc_score(labels, probas_)
        return auc
    else:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are 'fisher', 'euclidean', 'mahalanobis', 'lda', 'kl_divergence', 'js_divergence', 'svm_margin', and 'auc_svm'.")
    


# generate lobe line
def add_lobeline(topofile):
    topo = read_file(topofile,para=4)
    linefile = rf'./lobe/lobesline.png'
    line = read_file(linefile,para=4)
    lineindex = np.argwhere(line[:,:,-1])
    topo[lineindex[:,0],lineindex[:,1],:] = line[lineindex[:,0],lineindex[:,1],:]
    save_data(topofile,topo)
    return
def addLobe(modelDf):
    modelDf['x'] = (410 + 2100*modelDf['RIGHT']).astype(int)
    modelDf['y'] = (390 + 2100*modelDf['NOSE']).astype(int)
    lobefile = rf'./lobe/lobes.csv'
    lobedf = read_file(lobefile)
    llist = []
    for i,irow in modelDf.iterrows():
        l = (lobedf[(lobedf['y']==irow['y']) & (lobedf['x']==irow['x'])]['lobe']).values[0]
        llist.append(l)
    modelDf['lobe'] = llist
    return modelDf
def calcLobe(modelDf,lobeList,method):
    lobeValueDict = {l:modelDf['value'][modelDf['lobe']==l].tolist() for l in lobeList}
    if method=='sum':
        return {k:np.nansum(v) for k,v in lobeValueDict.items()}
    elif method=='mean':
        return {k:np.nanmean(v) for k,v in lobeValueDict.items()}
    elif method=='median':
        return {k:np.nanmedian(v) for k,v in lobeValueDict.items()}
    elif method=='list':
        return lobeValueDict
    else:
        raise
def plot_bar(valueDict,colorDict,figfile):
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    num = len(valueDict)
    x = list(valueDict.keys())
    y = list(valueDict.values())
    c = list(colorDict.values())
    ax.bar(x=x,height=y,color=c)
    ax.set_xlabel('')
    ax.set_xticks(list(range(num)))
    ax.set_xticklabels(x,fontsize=10,)
    ax.set_ylabel('')
    ax.set_yticks([0,1])
    ax.set_yticklabels([0,1],fontsize=10,)
    fig.subplots_adjust(bottom=0.12,top=0.98,left=0.12,right=0.98)
    save_data(figfile)
    return