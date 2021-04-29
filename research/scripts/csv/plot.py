import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import StrMethodFormatter
np.cat = np.concatenate
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
SMALL_SIZE = 10
#MEDIUM_SIZE = 18
#BIGGER_SIZE = 22
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

path = pathlib.Path('.')

envs = ['Luxo', 'Urchin']
modes = ['real','lenv']
tags = ['eplen', 'epret', 'succreal'][1:]
mode_map = {'real': 'Real Simulator', 'lenv': 'Learned Simulator'}
tag_map = {'eplen': 'Ep Length', 'epret': 'Ep Return (training)', 'succreal': 'Success Rate (on real)'}
smooth_map = {'eplen': 0.6, 'epret': 0.6, 'succreal': 0.8}

LuxoReal = dict(
EpRet=-2.5428032527156175,
EpLen=13.345999999999997,
success_rate=0.9990000000000001,
)

UrchinReal = dict(
EpRet=-7.8435451961323635,
EpLen=39.345,
success_rate=0.9260000000000003,
)

LuxoLenv = dict(
EpRet=-3.2117746651813395,
EpLen=19.345,
success_rate=0.9880000000000001,
)

UrchinLenv= dict(
EpRet=-9.303390638424455,
EpLen=47.555,
success_rate=0.8509999999999998,
)


def smooth(scalars, weight=0.8):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

for env in envs:
    files = []
    for tag in tags:
        realname = f'{env}_real_{tag}.csv'
        realfile = path / realname
        realdf = pd.read_csv(realfile)
        plt.plot(realdf.Step, realdf.Value, linewidth=2.0, color='C0', alpha=0.25)
        plt.plot(realdf.Step, smooth(realdf.Value, weight=smooth_map[tag]), linewidth=2.0, color='C0', label=mode_map['real'])

        lenvname = f'{env}_lenv_{tag}.csv'
        lenvfile = path / lenvname
        lenvdf = pd.read_csv(lenvfile)
        plt.plot(lenvdf.Step, lenvdf.Value, linewidth=2.0, color='C1', alpha=0.25)
        plt.plot(lenvdf.Step, smooth(lenvdf.Value, weight=smooth_map[tag]), linewidth=2.0, color='C1', label=mode_map['lenv'])

        plt.title(f'{env} {tag_map[tag]}')
        plt.xlabel('Environment steps')
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1g}')) # No decimal places
        plt.ylabel(tag_map[tag])
        plt.legend()
        fname = f'{env}_{tag}.png'
        files += [fname]
        plt.tight_layout()
        plt.savefig(fname)
        plt.clf()
    arr = [plt.imread(fname) for fname in files]
    arr = np.cat(arr, 1)
    plt.imsave(f'{env}_rl.png', arr)
    #plt.imshow(arr)
    #plt.title('Luxo')
    #plt.axis('off')
    #plt.savefig('test.png', bbox_inches='tight')
    #plt.show()


