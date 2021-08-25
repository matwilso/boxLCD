import numpy as np
from research import utils
import pathlib
import pickle

metrics = [
        #'F1',
        #'PRC',
        #'REC',
        'SSIM',
        'PSNR', 
        'COS*', 
        'FVD*', 
        ]
met_map = {
        'SSIM': 'ssim',
        'PSNR': 'psnr',
        'FVD*': 'fvd',
        'F1': 'f1',
        'PRC': 'precision',
        'REC': 'recall',
        'COS*': 'cosdist',
}
envs = ['Dropbox', 'Bounce', 'Bounce2', 'Object2', 'Urchin', 'Luxo', 'UrchinCube', 'LuxoCube']
models = ['RSSM', 'FRNLD']
mega = np.zeros([len(metrics), len(envs), len(models)])

sigfigs = {}
for i, model in enumerate(models):
    for j, env in enumerate(envs):
        path = pathlib.Path(f'logs/evalz/{model}/{env}')
        file = path / 'logger.pkl'
        with file.open('rb') as f:
            arr = pickle.load(f)
        test = utils.filtdict(arr, 'test:', fkey=lambda x:x[5:])
        testp = utils.filtdict(test, 'p:', fkey=lambda x:x[2:])
        for k, met in enumerate(metrics):
            std = testp[met_map[met]][1]
            #print(std, np.round(np.log(std)))
            sigfigs[met] = int(min(4, -np.round(np.log(std))))
            mega[k, j, i] = testp[met_map[met]][0]
            print(model, env, met,mega[k,j,i]) 

maxmega = mega * np.concatenate([np.ones_like(mega[:2]), -np.ones_like(mega[2:])], axis=0)

for i, row in enumerate(mega.reshape([-1,len(models)*len(envs)])):
    print(metrics[i], end=' ')
    mm = maxmega[i]
    sf = sigfigs[metrics[i]]
    for j,col in enumerate(row):
        post = str(sf) + 'f}'
        if mm.argmin(1)[j//2] == j%2:
            val = ('{col:.'+post).format(col=col)[:5]
        else:
            val = ('{col:.'+post).format(col=col)[:5]
            val = '\\textbf{' + val + '}'
        print('& '+val, end=' ')
    print('\\\\')

