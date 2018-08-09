import numpy as np
import os

data_dir = '/home/toledo/deeplearningWithPython/data/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print('HEADER: ', header)
print('DATA LEN: ', len(lines))

# parse data

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

def normalize(data):
    mean = data[:20000].mean(axis=0)
    data -= mean
    std = data[:20000].std(axis=0)
    data /= std
    return data

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):

    '''
    data--The original array of floating-point data
    lookback--How many timesteps back the input data should go
    delay--How many timesteps in the future the target should be
    min_index and max_index--Indices in the data array that delimit which
        timesteps to draw from.
    shuffle--whether to shuffle the samples or draw them in chronological order
    step--The peroid, in timesteps, at which you sample data.
    '''

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
