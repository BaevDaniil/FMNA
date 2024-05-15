import xarray as xr
import pandas as pd
import os
import re
import json


def loadMinianData(path, pattern, loadingData, meta):
    vpath = os.path.normpath(path)
    vlist = [x[0] for x in os.walk(vpath) if re.search(pattern, x[0])]

    sourceData = []
    for path in vlist:
        dataList = []
        for dataName in loadingData:
            data = xr.open_zarr(path + os.sep + dataName + '.zarr')
            data['animal'] = [ path.split(os.sep)[meta['animal']] ]
            data['session'] = [ path.split(os.sep)[meta['session']] ]
            dataList.append(data)
        sourceData.append(xr.merge(dataList))
    return sourceData

def loadData(path):
    vpath = os.path.normpath(path)
    vlist = os.listdir(vpath)
    
    metafile = next(filter(lambda x: x.endswith('json'), vlist))
    with open(path + os.sep + metafile) as f:
        meta = json.load(f)

    signals = []
    for m in meta:
        s = pd.read_csv(path + os.sep + m['file'], sep=';', index_col=0, header=0)
        signals.append((m, s))
    return signals
    
    