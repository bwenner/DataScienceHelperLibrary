"""Small Library for very common uses.

This module provides parameterizable functions for analysing data frames with log output.
"""

__version__ = '0.1'
__author__ = 'Benjamin Wenner'


from collections import defaultdict

import numpy as np
import pandas as pd

import os
from multiprocessing import Pool


def PrintLine(text, number = 20, character = '-'):
    print(character * number, text, character * number)

def ReadCsvFiles(files, delimiter = ','):
    '''
    INPUT:
    files: List of file names
    
    OUTPUT:
    dfs: overall status (bool) all succ, list of data frames
    '''
    dfs = []
    allsucc = True
    for file in files:
        try:
            print('reading', file)
            curdf = pd.read_csv(file, delimiter = delimiter)
            dfs.append(curdf)
        except Exception as e:
            print('Could not read file ', file, ': ', str(e))
            allsucc = False
            dfs.append(None)
    return allsucc, dfs 
    
def GetColumnsHavingNan(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    ret: returns list of columns having nan values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNan": df is None')
    return df.isnull()

def GetColumnsHavingNoNan(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    lst: returns list of columns having no nan values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNoNan": df is None')
    return df[~df.isnull().any()]
    #return df[~df.isnull().mean() == 0]

def GetColumnHavingNanPercent(df, percent):
    '''
    INPUT:
    df: Pandas Dataframe
    percent: element from [0,1]
    OUTPUT:
    lst: returns dataframe of columns having more than 0.x missing values
    '''
    if df is None:
        raise ValueError('Fnc "GetColumnsHavingNanPercent": df is None')
    if percent > 1 or percent < 0:
        raise ValueError('Fnc "GetColumnsHavingNanPercent": percent is out of bounds [0,1]: ', percent)
    return df[df.columns[df.isnull().mean() > percent]]

def GetPropAInGroupB(df, prop, group):
    '''
    INPUT:
    df: Pandas Dataframe
    prop: Column whose proportion we are looking for after grouping by column group
    group: Column the data will be grouped by to calc proportion for column prop
    
    OUTPUT:
    proportion: proportion of column prop in each group grouped by column group (for example group by employment
    status and get mean for job satisfaction)
    '''
    if df is None:
        raise ValueError('Fnc "GetPropAInGroupB": df is None')
    if not prop in df.columns:
        raise ValueError(str.format('Fnc "GetPropAInGroupB": prop: column "{}" not in dataframe', prop))
    if not group in df.columns:
        raise ValueError(str.format('Fnc "GetPropAInGroupB": group: column "{}" not in dataframe', group))
    return df.groupby(group).mean()[prop]

def RemoveAllRowsHavingAnyMissingValue(df, log = True):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe whose rows all have values
    '''
    if df is None:
        raise ValueError('Fnc "RemoveAllRowsHavingAnyMissingValue": df is None')
    colstoremove = GetColumnsHavingNan(df)
    if log:
        for remcol in colstoremove:
            print('Column will be removed from dataframe: ', remcol)
    return df.dropna()

def RemoveColumnsByPercent(df, percent):
    cols2remove = GetColumnHavingNanPercent(df, percent)
    if cols2remove.shape[0] > 0:
        print('Columns will be removed: ', list(cols2remove.columns))
    return df[[col for col in df.columns if col not in cols2remove]]

def RemoveRowsWithAllMissingValues(df, subset = None):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe without rows with all features = nan
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsWithAllMissingValues": df is None')
    return RemoveRowsByThresh(df, 1, subset)

def RemoveRowsByThresh(df, thresh, subset = None):
    '''
    INPUT:
    df: Pandas Dataframe
    thresh: Require that many non-NA values.
    
    OUTPUT:
    df: Dataframe with rows having at least 'thresh' values <> nan
    '''
    if df is None:
        raise ValueError('Fnc "RemoveRowsByThresh": df is None')
    if subset is not None and type(subset) is str:
        subset = [subset]
    return df.dropna(thresh = thresh, subset = subset)

def ApplyFillMissingValuesWithMean(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: Dataframe and nan values filled with mean
    '''
    if df is None:
        raise ValueError('Fnc "ApplyFillMissingValuesWithMean": df is None')
    fill_na = lambda x: x.fill_na(x.mean())
    df.apply(fill_na, axis = 0)
    return df

def SplitDataInXY(df, colx, coly):
    '''
    INPUT:
    df: Pandas Dataframe
    colx: list of columns for X
    coly: list of columns for y
    
    ret: data frames x, y splitted by colx, coly
    '''
    if df is None:
        raise ValueError('Fnc "SplitDataInXY": df is None')
    return df[colx], df[coly]

def SelectColumnsByType(df, typeinc, typeexc = None):
    '''
    INPUT:
    df: Pandas Dataframe
    typeinc: list of types to include
    typeexc: list of types to exclude (default none)
    
    ret: data frames x, y selected by colx, coly
    '''
    if df is None:
        raise ValueError('Fnc "SelectColumnsByType": df is None')
    if type(typeinc) is str:
        typeinc = [typeinc]
    return df.select_dtypes(include = typeinc, exclude = typeexc)

def SelectColumnsByTypeObject(df):
    '''
    INPUT:
    df: Pandas Dataframe
    
    OUTPUT:
    df: data frame with columns of type 'object'
    '''
    return SelectColumnsByType(df, ['object'])


def Apply10Encoding(df, col, newcol, vals):
    '''
    INPUT:
    df: Pandas Dataframe
    col: column to encde with 1 or 0
    newcol: new column name
    vals: values to encode with 1
    
    OUTPUT:
    df: data frame with columns of type 'object'
    '''
    if df is None:
        raise ValueError('Fnc "Apply10Encoding": df is None')
    _encode = lambda x: 1 if x in vals else 0 
    df[newcol] = df[col].apply(_encode)
    return df

def AnalyseNanColumns(df):
    if df is None:
        raise ValueError('Fnc "AnalyseNanColumns": df is None')
    PrintLine('Analysis of Columns with NaN values')
    dfnull = df.isnull().mean()
    
    if dfnull.shape[0] == 0:
        print('All columns have values')
        PrintLine('Analysis of Columns with NaN values finished')
        return
    tmp = dfnull[dfnull == 0]
    if tmp.shape[0] > 0:
        print('Columns having all values: ', tmp)
    tmp = dfnull[(dfnull > 0) & (dfnull <= 0.05)]
    if tmp.shape[0] > 0:
        print('Columns having > 0% and <= 5% missing values: ', tmp)
    tmp = dfnull[(dfnull > 0.05) & (dfnull <= 0.2)]
    if tmp.shape[0] > 0:
        print('Columns having > 5% and <= 20% missing values: ', tmp)
    tmp = dfnull[(dfnull > 0.2) & (dfnull <= 0.5)]
    if tmp.shape[0] > 0:
        print('Columns having > 20% and <= 50% missing values: ', tmp)
    tmp = dfnull[(dfnull > 0.5) & (dfnull <= 0.7)]
    if tmp.shape[0] > 0:
        print('Columns having > 50% and <= 70% missing values: ', tmp)
    tmp = dfnull[dfnull > 0.7]
    if tmp.shape[0] > 0:
        print('Columns having > 70% missing values: ', tmp)
    PrintLine('Analysis of Columns with NaN values finished')
    
def AnalyseValueCounts(df, columns = None, types = None, considerMaxValues = 20):
    if df is None:
        raise ValueError('Fnc "AnalyseValueCounts": df is None')
    if (considerMaxValues < 0 or considerMaxValues > 30):
        raise ValueError('Fnc "AnalyseValueCounts": considerMaxValues < 0 or too large (> 30)', considerMaxValues)
    logtxt = 'Considering columns: '
    if columns is None or types is None:
        if columns is None and types is None:
            columns = list(df.columns)
        elif object is None:
            columns = list(df.columns)
        elif columns is None:
            columns = list(SelectColumnsByType(df, types).columns)
    if len(columns) == 0:
        print('No columns to analyse value counts for. Passed columns and types: ', columns, types)
        return
    print(logtxt, columns)
    PrintLine('Dataframe value counts analye started')
    for col in columns:
        PrintLine('', character = '*')
        vcser = df[col].value_counts()
        if vcser.shape[0] > considerMaxValues:
            print('Column "{}" has more that {} different values: '.format(col, considerMaxValues), vcser.shape[0])
        else:
            print(vcser)
        PrintLine('', character = '*')
    PrintLine('Dataframe value counts analye finished')
    
    
    
def AnalyseDataframe(df):
    if df is None:
        raise ValueError('Fnc "AnalyseDataframe": df is None')
    PrintLine('Dataframe analysis started')
    print('Shape = ', df.shape)
    AnalyseNanColumns(df)
    
    
    
    
    
    PrintLine('Dataframe analysis finished')








