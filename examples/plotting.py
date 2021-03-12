import yaml
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


log = pathlib.Path('./logs/paper/')
names = []
name_order = ['Dropbox', 'Bounce', 'Urchin', 'UrchinBall', 'UrchinBalls', 'UrchinCubes']
order = ['3_128_4', '3_256_8', '3_512_16']
imgs = []
for name in name_order:
  name = log / name
  for o in order:
    csv = (name / o) / 'logger.csv'
    #hps = (name / o) / 'hps.yaml'
    #with hps.open('r') as f:
    #  arr = yaml.load(f, Loader=yaml.Loader)
    if not csv.exists():
      continue
    out = pd.read_csv(csv)
    test_loss = out.test_loss.array.to_numpy()[:-1]
    train_loss = out.loss.array.to_numpy()[:-1]
    num_vars = out.num_vars.array.to_numpy()[:-1]
    x = np.arange(len(test_loss)) + 1
    plt.plot(x, test_loss, label=f'{num_vars[0]:.2e} params', linewidth=2)
  plt.title(name.name)
  plt.legend()
  #plt.xlim(-5, 35)
  #plt.ylim(0, 0.10)
  plt.xlabel('Epochs')
  plt.ylabel('Test Loss (bits/dim)')
  plt.tight_layout()

  fig = plt.gcf()
  fig.canvas.draw()
  width, height = (fig.get_size_inches() * fig.get_dpi()).astype(np.int)
  img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
  imgs += [img]
  plt.savefig(name.name + '.png')
  plt.clf()
  # plt.show()

cat = np.concatenate

r1 = cat(imgs[:3], 1)
r2 = cat(imgs[3:], 1)
all = cat([r1, r2], 0)
plt.imsave('All.png', all)
