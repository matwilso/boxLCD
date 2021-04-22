import numpy as np
import matplotlib.pyplot as plt
import time
from boxLCD import envs
env = envs.Dropbox()
env.seed(8)
print(env.G.ep_len)
obs = env.reset()
np.random.seed(0)
#env.seed(8)

l = []
p = []
def func(env):
    env.reset()
    for i in range(5):
        obs, _, done, info = env.step(env.action_space.sample())
    pl = env.render(mode='human', return_pyglet_view=True)
    p, l = pl[:,:env.viewer.width-1], pl[:,1+env.viewer.width:]
    return p,l

#pls = [func(env) for env in [envs.Dropbox(), envs.Bounce(), envs.Bounce2(), envs.Object2()]]
pls = [func(env) for env in [envs.Urchin(), envs.Luxo(), envs.UrchinCube(), envs.LuxoCube(), envs.UrchinBall(), envs.LuxoBall()]]
p,l = zip(*pls)

def linecat(arr, axis=0):
    if axis == 1:
        zero = np.zeros_like(arr[0])[:,:1]
    elif axis == 0:
        zero = np.zeros_like(arr[0])[:1]
    new_arr = []
    for a in arr:
        new_arr += [a, zero]
    new_arr = new_arr[:-1]
    return np.concatenate(new_arr, axis=axis)


plt.imsave('tier1.png', linecat([linecat(p,1), linecat(l,1)]))
#plt.imsave('dropbox.png', np.concatenate([np.concatenate(p, 1), np.concatenate(l, 1)]))
#plt.imsave('dropbox.png', 255*np.concatenate(p, 1).astype(np.uint8))
exit()


def write_gif(name, frames, fps=60):
  start = time.time()
  from moviepy.editor import ImageSequenceClip
  # make the moviepy clip
  clip = ImageSequenceClip(list(frames), fps=fps)
  clip.write_gif(name, fps=fps)
  #copyfile(name, str(pathlib.Path(f'~/Desktop/{name}').expanduser()))
  print(time.time() - start)

imgs = []
while True:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
    imgs += [env.render(mode='human', return_pyglet_view=True)]
    print(obs['lcd']*1.0, '\n')
    time.sleep(0.01)
    if done:break
#write_gif('urchin_cubes.gif', imgs)

