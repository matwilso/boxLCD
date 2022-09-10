import pathlib
import pickle

import numpy as np

from research import utils

metrics = [
    'FVD',
    #'F1',
    #'PRC',
    #'REC',
    'SSIM',
    'PSNR',
    'LPIPS',
]
met_map = {
    'SSIM': 'ssim',
    'PSNR': 'psnr',
    'FVD': 'fvd',
    'F1': 'f1',
    'PRC': 'precision',
    'REC': 'recall',
    'LPIPS': 'cosdist',
}
envs = [
    'Dropbox',
    'Bounce',
    'Bounce2',
    'Object2',
    'Urchin',
    'Luxo',
    'UrchinCube',
    'LuxoCube',
]
models = ['RSSM', 'FIT']
mega = np.zeros([len(metrics), len(envs), len(models)])

sigfigs = {}
for i, model in enumerate(models):
    for j, env in enumerate(envs):
        path = pathlib.Path(model + '_' + env)
        file = path / 'logger.pkl'
        with file.open('rb') as f:
            arr = pickle.load(f)
        test = utils.filtdict(arr, 'test:', fkey=lambda x: x[5:])
        testp = utils.filtdict(test, 'p:', fkey=lambda x: x[2:])
        for k, met in enumerate(metrics):
            std = testp[met_map[met]][1]
            # print(std, np.round(np.log(std)))
            sigfigs[met] = int(min(4, -np.round(np.log(std))))
            mega[k, j, i] = testp[met_map[met]][0]
            print(model, env, met, mega[k, j, i])

for i, row in enumerate(mega.reshape([-1, len(models) * len(envs)])):
    print(metrics[i], end=' ')
    sf = sigfigs[metrics[i]]
    for col in row:
        post = str(sf) + 'f}'
        val = ('{col:.' + post).format(col=col)[:5]
        print('& ' + val, end=' ')
    print('\\\\')
