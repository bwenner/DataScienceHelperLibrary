"""Small Library for very common uses.

This module provides parameterizable functions for analysing data frames with log output.
"""

__version__ = '0.1'
__author__ = 'Benjamin Wenner'


from collections import defaultdict

import numpy as np
import pandas as pd
import fnmatch as fnmatch

import os
from multiprocessing import Pool


def IsMatch(txt, wildcard):
    return fnmatch.fnmatch(txt, wildcard)

def PrintLine(text = '', number = 20, character = '-'):
    print(character * number, text, character * number)

def ReadCsvFiles(files, delimiter = ','):
    '''
    INPUT:
    files: List of file names
    
    OUTPUT:
    dfs: overall status (bool) all succ, list of data frames
    '''
    if files is None or len(files) < 0:
        raise ValueError('Fnc "ReadCsvFiles": files is None or empty')
    if type(files) is str:
        files = [files]
    dfs = []
    allsucc = True
    PrintLine('Start reading files')
    for file in files:
        try:
            print('Reading: ', file)
            curdf = pd.read_csv(file, delimiter = delimiter)
            dfs.append(curdf)
        except Exception as e:
            print('Could not read file ', file, ': ', str(e))
            allsucc = False
            dfs.append(None)
    log = 'Reading files successfully finished'
    if not allsucc:
        log = 'Reading files finished with errors'
    PrintLine(log)
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

def SplitDataByValuesInColumn(df, column, values):
    '''
    INPUT:
    df: Pandas Dataframe
    column: Column name
    values: list of list of values. For example: [ [val1_A, val2_A], [val1_B, val2_B] ] 
    results in two dataframes. Values of passed column are in [val1_A, val2_A] for first data frame 
    and values for passed column are in [val1_B, val2_B] for seccond dataframe.
    '''
    if df is None:
        raise ValueError('Fnc "SplitDataByValuesInColumn": df is None')
    dflist = []
    for valueList in values:
        newdf = df[df[column].isin(valueList)]
        dflist.append(newdf)
    return dflist
    
def SelectColumnsByWildcard(df, wildcards):
    '''
    INPUT:
    df: Dataframe
    wildcards: string or list of strings
    
    output: Dataframe with columns
    '''
    if df is None:
        raise ValueError('Fnc "SelectColumnsByWildcard": df is None')
    if wildcards is None or len(wildcards) < 1:
        raise ValueError('Fnc "SelectColumnsByWildcard": No wildcards passed: ', wildcards)
    rem = []
    if type(wildcards) is str:
        wildcards = [wildcards]
    allColumns = list(df.columns)
    PrintLine('Start finding and keeping columns matchting to wildcards: '.format(wildcards))
    keep = []
    for col in allColumns:
        for wc in wildcards:
            if IsMatch(col, wc):
                keep.append(col)
                break
    if len(keep) == 0:
        print('No column names found matchting to wildcards')
    else:
        print('Columns found to keep: ', keep)
    PrintLine('Finished keeping columns matchting to wildcards')
    return df[ keep ]
    
    
def RemoveColumnsByWildcard(df, wildcards):
    '''
    INPUT:
    df: Dataframe
    wildcards: string or list of strings
    
    output: Dataframe without columns
    '''
    if df is None:
        raise ValueError('Fnc "RemoveColumnsByWildcard": df is None')
    if wildcards is None or len(wildcards) < 1:
        raise ValueError('Fnc "RemoveColumnsByWildcard": No wildcards passed: ', wildcards)
    rem = []
    if type(wildcards) is str:
        wildcards = [wildcards]
    allColumns = list(df.columns)
    PrintLine('Start finding and removing columns matchting to wildcards: '.format(wildcards))
    for col in allColumns:
        for wc in wildcards:
            if IsMatch(col, wc):
                rem.append(col)
                break
    keep = [ ac for ac in allColumns if ac not in rem ]
    if len(rem) == 0:
        print('No column names found matchting to wildcards')
    else:
        print('Columns found to remove: ', rem)
    PrintLine('Finished removing columns matchting to wildcards')
    return df[ keep ]
    
def RemoveColumnsByPercent(df, percent):
    if df is None:
        raise ValueError('Fnc "RemoveColumnsByPercent": df is None')
    cols2remove = GetColumnHavingNanPercent(df, percent)
    if cols2remove.shape[0] > 0:
        print('Columns will be removed: ', list(cols2remove.columns))
    return df[[col for col in df.columns if col not in cols2remove]]

def RemoveColumnsHavingOnlyOneValue(df):
    if df is None:
        raise ValueError('Fnc "RemoveColumnsHavingOnlyOneValue": df is None')
    PrintLine('Start searching and removing columns with one value:')
    cols = []
    keep = []
    for col in list(df.columns):
        vc = df[col].value_counts()
        if vc.shape[0] == 1 and vc[vc.index[0]] == df.shape[0]:
            print('Removing {} - "{}"'.format(col, vc.index[0]))
            cols.append(col)
            continue
        keep.append(col)
    dfret = df[keep]
    PrintLine('finished searching and removing columns with one value:')
    return dfret
    
    
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

def RemoveRowsWithValueInColumn(df, column, values):
    if df is None:
        raise ValueError('Fnc "RemoveRowsWithValueInColumn": df is None')
    if type(values) is str:
        values = [values]
    dfret = df[~df[column].isin(values)]
    print('{} rows (ca. {}%) have been removed with value/s "{}" in column "{}"'.format(df.shape[0] - dfret.shape[0], "{0:.2f}".format((df.shape[0] - dfret.shape[0]) * 100 / df.shape[0]), values, column))
    return df[~df[column].isin(values)]
    
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
    
    OUTPUT:
    returns data frames x, y selected by colx, coly
    
    NOTES:
    To select all numeric types, use np.number or 'number'
    To select strings you must use the object dtype, but note that this will return all object dtype columns
    See the numpy dtype hierarchy
    To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
    To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'
    To select Pandas categorical dtypes, use 'category'
    To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or 'datetime64[ns, tz]'
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


def Apply10Encoding(df, col, vals, newcol = None, drop = True):
    '''
    INPUT:
    df: Pandas Dataframe
    col: column to encde with 1 or 0
    vals: values to encode with 1
    newcol: new column name
    drop: if newcol provided, col will be dropped.
    
    OUTPUT:
    df: data frame with columns of type 'object'
    '''
    if df is None:
        raise ValueError('Fnc "Apply10Encoding": df is None')
    if type(vals) is str:
        vals = [vals]
    _encode = lambda x: 1 if x in vals else 0 
    if newcol is None:
        newcol = col
        drop = False
    df[newcol] = df[col].apply(_encode)
    if drop:
        return df.dropna()
    return df

def Apply10EncodingOnColumnWithMultiValuesInCell(df, column, values, drop = True):
    if df is None:
        raise ValueError('Fnc "Apply10Encoding": df is None')
    if df[column].dtype != 'O':
        raise ValueErro('Fnc "Apply10Encoding": Invalid dtype "{}" for column "{}""'.format(df[columns].dtype, column))
    dfcopy = df.copy(deep = True)
    for val in values:
        newcol = column + '_' + val
        dfcopy[newcol] = dfcopy[column].apply(lambda x: 1 if val in x else 0)
    if drop == True:
        dfcopy = dfcopy.drop(column, axis = 1)
    return dfcopy    
    
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
        print('Columns having all values: ')
        print(tmp)
    tmp = dfnull[(dfnull > 0) & (dfnull <= 0.05)]
    if tmp.shape[0] > 0:
        print('Columns having > 0% and <= 5% missing values: ')
        print(tmp)
    tmp = dfnull[(dfnull > 0.05) & (dfnull <= 0.2)]
    if tmp.shape[0] > 0:
        print('Columns having > 5% and <= 20% missing values: ')
        print(tmp)
    tmp = dfnull[(dfnull > 0.2) & (dfnull <= 0.5)]
    if tmp.shape[0] > 0:
        print('Columns having > 20% and <= 50% missing values: ')
        print(tmp)
    tmp = dfnull[(dfnull > 0.5) & (dfnull <= 0.7)]
    if tmp.shape[0] > 0:
        print('Columns having > 50% and <= 70% missing values: ')
        print(tmp)
    tmp = dfnull[dfnull > 0.7]
    if tmp.shape[0] > 0:
        print('Columns having > 70% missing values: ')
        print(tmp)
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
            print('More than {} different values: '.format(considerMaxValues), vcser.shape[0])
            print('Name: ', col, ', dtype: ', vcser.dtype)
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

def ConvertColumnToType(df, columns, newtype = 'float64', replace = None):
    '''
    INPUT:
    df: Pandas Dataframe
    columns: Column/s to convert
    newtype: Final type
    replace: Dictionary like { '': { "rep1", "rep2", ...} }
    
    OUTPUT:
    Dataframe with converted column
    '''
    if df is None:
        raise ValueError('Fnc "ConvertColumnToType": df is None')
    dfcopy = df.copy(deep = True)
    if type(columns) is str:
        columns = [columns]
    PrintLine('Start replacing and converting columns')
    for col in columns:
        if replace is not None:
            if type(replace) is not dict:
                raise ValueError('Fnc "ConvertColumnToType": type of replace is no dictionary: ', replace)
            for key in replace.keys():
                if df[col].dtype == newtype:
                    print('Column "{}" dtype is already {}'.format(col, newtype))
                    break
                val = replace[key]
                print(key, val)
                for repval in val:
                    print('Replacing "{}" with "{}"'.format(repval, key))
                    dfcopy[col] = dfcopy[col].apply(lambda x: x.replace(repval, key).strip())
        dfcopy[col] = dfcopy[col].astype(newtype)    
    PrintLine('Replacing and converting columns finished')
    return dfcopy

def CleanValuesInColumn(df, columns, trim = True, clean = None):
    '''
    INPUT:
    df: Dataframe
    columns: column name or list of columns
    trim: remove leading and trailing spaces
    clean: dictionary<string, list<string>>: replace any string value in list with key
    
    OUTPUT:
    Dataframe with cleaned column/s
    '''
    if df is None:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": df is None')
    dfcopy = df.copy(deep = True)
    if type(columns) is str:
        columns = [columns]
    PrintLine('Start cleaning values in columns:')
    for col in columns:
        if dfcopy[col].dtype == 'O':
            applied = ''
            if clean is not None:
                for key in clean.keys():
                    repvalues = clean[key]
                    print('current key "{}", current repvals = "{}"'.format(key, repvalues))
                    if len(repvalues) == 1:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key, regex = False)
                    elif len(repvalues) == 2:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key, regex = False).str.replace(repvalues[1], key, regex = False)
                    elif len(repvalues) == 3:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key, regex = False).str.replace(repvalues[1], key, regex = False).str.replace(repvalues[2], key, regex = False)
                    elif len(repvalues) == 4:
                        dfcopy[col] = dfcopy[col].str.replace(repvalues[0], key, regex = False).str.replace(repvalues[1], key, regex = False).str.replace(repvalues[2], key, regex = False).str.replace(repvalues[3], key, regex = False)
                    else:
                        print('Number of replace values not supported: ', len(repvalues))
            if trim:
                dfcopy[col] = dfcopy[col].apply(lambda x: x.strip())
        else:
            print('Cannot apply cleaning on column "{}", dtype is: '.format(col, dfcopy[col].dtype))
    return dfcopy
    
def GetUniqueValuesListFromColumn(df, column, trim = False, clean = None, splitby = None, asc = None, ignoreempty = False):
    if df is None:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": df is None')
    if type(column) is not str:
        raise ValueError('Fnc "GetUniqueValuesFromColumn": column is not string: "', column, '"')
    dfcopy = None
    if trim or clean is not None:
        dfcopy = CleanValuesInColumn(df, column, trim, clean)
    else:
        dfcopy = df.copy(deep = True)
    vals = dfcopy[column].unique()
    finalVals = []
    if splitby is None:
        finalVals = vals
    elif dfcopy[column].dtype == 'O':
        for val in vals:
            splitted = val.split(splitby)
            for splitvalue in splitted:
                if splitvalue in finalVals:
                    continue
                finalVals.append(splitvalue)
                print('Appended: ', splitvalue)
        if trim:
            for ind in range(len(finalVals)):
                finalVals[ind] = finalVals[ind].strip()
    if ignoreempty:
        finalVals = [x for x in finalVals if type(x) is str and len(x) > 0]
    if asc is not None:
        if type(asc) is bool:
            finalVals.sort(reverse = asc == False)
    return finalVals
    
    
'''
def SplitCategoricalValues(df, dirValTypes):
    categoricalColumns, needsToEncodeBin, needsToEncodeMulti, needsToEncodeStringBin, needsToEncodeStringMulti, ignoreColumns4Encoding = {}, {}, {}, {}, {}, []

    for col in dirValTypes['categorical']:
        if not col in df:
            continue

        dfind = df.columns.get_loc(col)

        valueCounts = df.iloc[:, dfind].value_counts()
        valCount = valueCounts.count()

        categoricalColumns[col] = valCount

        added = False
        for ax in valueCounts.axes:
            for val in ax.values:
                try:
                    nbr = int(val)

                    # if any other value appears it must be new encoded
                    if nbr == 0 or nbr == 1:
                        continue

                    if valCount == 2:
                        if not col in needsToEncodeBin:
                                needsToEncodeBin[col] = []
                        if (val in needsToEncodeBin[col]):
                            continue
                        needsToEncodeBin[col].append( val )
                    else:
                        if not col in needsToEncodeMulti:
                                needsToEncodeMulti[col] = []
                        if (val in needsToEncodeMulti[col]):
                            continue
                        needsToEncodeMulti[col].append( val )
                    added = True
                except:
                    if valCount == 2:
                        if not col in needsToEncodeStringBin:
                                needsToEncodeStringBin[col] = []
                        if (val in needsToEncodeStringBin[col]):
                            continue
                        needsToEncodeStringBin[col].append( val )
                    else:
                        if not col in needsToEncodeStringMulti:
                                needsToEncodeStringMulti[col] = []
                        if (val in needsToEncodeStringMulti[col]):
                            continue
                        needsToEncodeStringMulti[col].append( val )
                    added = True
        if not added:
            ignoreColumns4Encoding.append(col)

    return categoricalColumns, needsToEncodeBin, needsToEncodeMulti, needsToEncodeStringBin, needsToEncodeStringMulti, ignoreColumns4Encoding    

'''




