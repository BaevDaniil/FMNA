import pandas as pd
import os
import json
import sys

def _createMeta(data):
    dataName = list(data.keys())[0]
    animal = data['animal'].values[0]
    session = data['session'].values[0]
    frame = data['frame'].count().values.item()
    neuron = data['unit_id'].count().values.item()
    file = animal + '_' + session + '_' + dataName + '.csv'
    return {'data': dataName,
            'animal': animal,
            'session': session,
            'frame': frame,
            'neuron': neuron,
            'file': file}

def _normSignals(signals):
    min = signals.min()
    max = signals.max()
    return (signals - min) / (max - min)

def preprocessSignals(data, normalize = False):
    signalsWithMeta = []
    for dataset in data:
        signals = pd.DataFrame(dataset.C).set_index((pd.DataFrame(dataset.C.unit_id)[0].rename("unit_id"))).T
        meta = _createMeta(dataset)
        if normalize:
           signalsWithMeta.append((meta, _normSignals(signals)))
        else:
           signalsWithMeta.append((meta, signals))
    return signalsWithMeta

def saveSignals(signals, savePath):
    meta = []
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for session in signals:
        meta.append(session[0])
        session[1].to_csv(savePath + os.sep + session[0]['file'], sep=';')
    metaJSON = meta
    with open(savePath + os.sep + 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(metaJSON, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    import dataloader

    with open(sys.argv[1]) as f:
        settings = json.load(f)

    data = dataloader.loadMinianData(settings['path'], settings['pattern'], settings['loading_data'], settings['meta'])
    signals = preprocessSignals(data, True)
    saveSignals(signals, settings['save_path'])
        