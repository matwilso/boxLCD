import itertools

import Box2D
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    frictionJointDef,
    polygonShape,
)
from pyglet.gl import glClearColor

import boxLCD.utils
from boxLCD import env_map, envs
from research import define_config, utils

KEY = pyglet.window.key
A = boxLCD.utils.A


def write_video(name, seed=7):
    env = env_map[name]()
    env.seed(seed)
    if name == 'LuxoCube':
        np_random = np.random.RandomState(3)
    else:
        np_random = np.random.RandomState(4)
    start = env.reset()
    env.render(mode='human')

    imgs = []
    for i in range(env.G.ep_len):
        action = np_random.uniform(-1, 1, env.action_space.shape[0])
        obs, rew, done, info = env.step(action)
        out = env.render(mode='human', return_pyglet_view=True)
        imgs += [out]
    utils.write_gif(name + '.gif', imgs, fps=10)


write_video('Object2', seed=1)
# for name in ['Dropbox', 'Bounce', 'Bounce2', 'Object2']:
#    write_video(name)
#
# for name in ['Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']:
#    write_video(name)
