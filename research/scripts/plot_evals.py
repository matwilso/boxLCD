#!/usr/bin/env python3
import argparse
import numpy as np
from collections import defaultdict
import pathlib
import pickle
import matplotlib.pyplot as plt

def main(args):
    evaldir = pathlib.Path(args.evaldir)
    models = list(evaldir.iterdir())
    metrics = defaultdict(lambda: defaultdict(lambda: {}))
    for model in models:
        for env in model.iterdir():
            path = env / 'logger.pkl'
            with path.open('rb') as f:
                metrics[env.name][model.name] = pickle.load(f)

    namemap = defaultdict(lambda: '')
    namemap.update(**{
        'ssim': '⬆️',
        'psnr': '⬆️',
        'f1': '⬆️',
        'fvd': '⬇',
        'cosdist': '⬇️',
        })
    imgs = []
    order = ['Dropbox', 'Bounce', 'Bounce2', 'Object2', 'Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']
    for env in order:
        fig, axes = plt.subplots(2,2)
        for i, model in enumerate(metrics[env]):
            for j, key in enumerate(['test:p:ssim', 'test:p:psnr', 'test:p:cosdist','test:p:fvd']):
            #for j, key in enumerate(['test:p:ssim', 'test:p:cosdist', 'test:p:f1','test:p:fvd']):
                name = key.split(':')[-1]
                axes.flat[j].set_title(name.upper() + ' ' + namemap[name])
                mean, std = metrics[env][model][key]
                # bar plot with standard deviation on jth axis
                axes.flat[j].bar(i, mean, yerr=std, capsize=5, label=f'{model}')
                ylim = axes.flat[j].get_ylim()[1]
                axes.flat[j].text(i, mean, f'{mean:.3f}', ha='center', va='bottom')
                axes.flat[j].spines['top'].set_visible(False)
                axes.flat[j].spines['right'].set_visible(False)
                #plt.bar(i, mean, yerr=std, label=f'{model} {key}')
                legend = axes.flat[j].legend()
                #legend.get_frame().set_alpha(0.1)

            #plt.legend()
        plt.suptitle(env, fontweight='bold')
        fig.tight_layout()
        #plt.savefig(f'test-{env}.png')
        fig.set_dpi(100)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs += [data]
        #plt.show()
        plt.close()
    imgs = np.array(imgs)
    imgs = imgs.reshape([2,5,*imgs.shape[1:]])
    # tranpose images to be in a 2x5 grid
    imgs = np.transpose(imgs, (0,2,1,3,4))
    # add borders between images
    imgs = np.pad(imgs, ((0,0),(0,10),(0,0),(0,10),(0,0)), mode='constant', constant_values=0)
    imgs = imgs.reshape([2*imgs.shape[1], -1, 3])
    plt.imsave('test.png', imgs)
    #plt.imshow(imgs)
    #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaldir', default='logs/evalz/')
    args = parser.parse_args()
    main(args)