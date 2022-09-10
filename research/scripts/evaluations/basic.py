import matplotlib.pyplot as plt
import numpy as np

from boxLCD import env_map, envs

env = envs.Dropbox()
env.seed(8)
print(env.G.ep_len)
obs = env.reset()
np.random.seed(0)
# env.seed(8)

l = []
p = []


def func(name):
    env = env_map[name]({'wh_ratio': 1.5})
    if name == 'UrchinCube':
        seed = 6
    elif name == 'Urchin':
        seed = 7
    elif name == 'LuxoCube':
        seed = 9
    elif name == 'Object2':
        seed = 7
    elif name == 'Bounce':
        seed = 3
    else:
        seed = 5
    env.seed(seed)
    env.reset()
    if 'Luxo' in name:
        N = 10
        np_random = np.random.RandomState(5)
        if 'LuxoCube' == name:
            np_random = np.random.RandomState(7)
        if 'Luxo' == name:
            np_random = np.random.RandomState(6)
    else:
        N = 5
        np_random = np.random.RandomState(3)
    if name in TIER0:
        N = 2
    for i in range(N):
        action = np_random.uniform(-1, 1, env.action_space.shape[0])
        obs, _, done, info = env.step(action)
    pl = env.render(mode='human', return_pyglet_view=True)
    p, l = pl[:, : env.viewer.width - 1], pl[:, env.viewer.width + 1 :]
    return p, l


TIER0 = ['Dropbox', 'Bounce', 'Bounce2', 'Object2']
TIER1 = ['Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']

# pls = [func(name) for name in TIER0]
pls = [func(name) for name in TIER1]
p, l = zip(*pls)


def linecat(arr, axis=0):
    if axis == 1:
        zero = np.zeros_like(arr[0])[:, :1]
    elif axis == 0:
        zero = np.zeros_like(arr[0])[:1]
    new_arr = []
    for a in arr:
        new_arr += [a, zero]
    new_arr = new_arr[:-1]
    return np.concatenate(new_arr, axis=axis)


# plt.imsave('tier0.png', linecat([linecat(p,1), linecat(l,1)]))
plt.imsave('tier1-extra.png', linecat([linecat(p, 1), linecat(l, 1)]))
# plt.imsave('dropbox.png', np.concatenate([np.concatenate(p, 1), np.concatenate(l, 1)]))
# plt.imsave('dropbox.png', 255*np.concatenate(p, 1).astype(np.uint8))
exit()


# def write_gif(name, frames, fps=60):
#  start = time.time()
#  from moviepy.editor import ImageSequenceClip
#  # make the moviepy clip
#  clip = ImageSequenceClip(list(frames), fps=fps)
#  clip.write_gif(name, fps=fps)
#  #copyfile(name, str(pathlib.Path(f'~/Desktop/{name}').expanduser()))
#  print(time.time() - start)
#
# imgs = []
# while True:
#    action = env.action_space.sample()
#    obs, _, done, info = env.step(action)
#    imgs += [env.render(mode='human', return_pyglet_view=True)]
#    print(obs['lcd']*1.0, '\n')
#    time.sleep(0.01)
#    if done:break
##write_gif('urchin_cubes.gif', imgs)
