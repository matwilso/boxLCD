#!/usr/bin/env python3
import argparse
import numpy as np
from collections import defaultdict
import pathlib
import pickle
import matplotlib.pyplot as plt

def plot(metrics):
    namemap = defaultdict(lambda: '')
    namemap.update(**{
        'ssim': '⬆️',
        'psnr': '⬆️',
        'f1': '⬆️',
        'fvd': '⬇',
        'cosdist': '⬇️',
        'action_log_mse': '⬇️',
        })
    imgs = []
    order = ['Dropbox', 'Bounce', 'Bounce2', 'Object2', 'Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']
    metric_names = ['test:p:ssim', 'test:p:action_log_mse', 'test:p:cosdist','test:p:fvd']
    for env in order:
        fig, axes = plt.subplots(2,2)
        for i, model in enumerate(metrics[env]):
            #for j, key in enumerate(['test:p:ssim', 'test:p:psnr', 'test:p:cosdist','test:p:fvd']):
            for j, key in enumerate(metric_names):
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

def print_out(metrics):
    """
    metrics is a dict of dicts of dicts with first key being env name, second key being model name, and third key being metric name
    the env names are 'Dropbox', 'Bounce', 'Bounce2', 'Object2', 'Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall'
    the modelnames are RSSM and FRNLD
    the metric names are 'test:p:ssim', 'test:p:action_log_mse', 'test:p:cosdist','test:p:fvd', but just the end part should be printed

    this function prints out something like this:
    & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD & RSSM & FRNLD \\
    FVD & 35.1 & 1.6 & 72.2 & 0.1 & 253.8 & 5.7 & 130.4 & 20.4 & 263.4 & 2.3 & 307.2 & 78.2 & 311.9 & 5.7 & 394.7 & 101.6 \\
    SSIM & 0.795 & 0.920 & 0.808 & 0.977 & 0.652 & 0.837 & 0.574 & 0.669 & 0.716 & 0.726 & 0.653 & 0.657 & 0.614 & 0.626 & 0.531 & 0.531 \\
    PSNR & 31.04 & 51.67 & 47.58 & 86.33 & 20.41 & 41.31 & 12.39 & 14.70 & 13.30 & 13.52 & 11.79 & 11.85 & 11.21 & 11.42 & 9.802 & 9.761 \\
    COS & 0.210 & 0.082 & 0.270 & 0.023 & 0.420 & 0.190 & 0.555 & 0.443 & 0.659 & 0.297 & 0.677 & 0.412 & 0.644 & 0.360 & 0.653 & 0.467 \\
    ```
    """
    env_names = ['Dropbox', 'Bounce', 'Bounce2', 'Object2', 'Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']
    model_names = ['RSSM', 'FRNLD']
    metric_names = ['test:p:ssim', 'test:p:psnr', 'test:p:cosdist','test:p:fvd']
    for metric in metric_names:
        name = metric.split(':')[-1].upper()
        print(name, end=' & ')
        for env in env_names:
            for model in model_names:
                print(f'{metrics[env][model][metric][0]:.3f}, end=' & ')
        print(f'\\\\')







def main(args):
    evaldir = pathlib.Path(args.evaldir)
    models = list(evaldir.iterdir())
    metrics = defaultdict(lambda: defaultdict(lambda: {}))
    for model in models:
        for env in model.iterdir():
            path = env / 'logger.pkl'
            with path.open('rb') as f:
                metrics[env.name][model.name] = pickle.load(f)
    print_out(metrics)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaldir', default='logs/evalz/')
    args = parser.parse_args()
    main(args)