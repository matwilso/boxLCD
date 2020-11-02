import time
import argparse
import itertools
from collections import defaultdict
import copy
import sys, math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import utils
from envs.box2d import B2D, World, Agent, Object
import pyglet
KEY = pyglet.window.key
A = utils.A

class Box(B2D):
    def __init__(self, cfg):
        #w = World(agents=[Agent('crab0')], objects=[])
        #w = World(agents=[Agent('crab0'), Agent('crab1'), Agent('crab2')], objects=[Object('object0'), Object('object1'), Object('object2')])
        #w = World(agents=[Agent('crab0')], objects=[Object('object0'), Object('object1'), Object('object2')])
        #w = World(agents=[], objects=[Object('object0')])
        w = World(agents=[Agent('crab0')], objects=[Object('object0')])
        super().__init__(w, cfg)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cfg import define_cfg, args_type, env_fn, make_env
    parser = define_cfg()
    parser.set_defaults(**{'exp_name': 'collect'})
    cfg = parser.parse_args()
    env = Box(cfg)
    env.reset()
    env.render()
    key_handler = KEY.KeyStateHandler()
    env.viewer.window.push_handlers(key_handler)
    window = env.viewer.window

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        print()
        print(x,y,dx,dy, buttons, modifiers)
        cp = env.SCALE*env.dynbodies['crab0:root'].position
        if np.linalg.norm(A[cp] - A[x,y]) < 100:
            env.dynbodies['crab0:root'].position += (dx/env.SCALE,dy/env.SCALE)

    paused = False
    traj = []
    past_keys = {}
    delay = 0.1*1/4
    dor = True
    plotting = False

    while True:
        action = env.action_space.sample()
        action = np.zeros_like(action)
        act_dict = env.get_act_dict(action)
        curr_keys = defaultdict(lambda: False)
        curr_keys.update({key: val for key, val in key_handler.items()})

        if curr_keys[KEY._0] or curr_keys[KEY.NUM_0]: 
            env.reset()
            time.sleep(0.1)
            traj = []
        if curr_keys[KEY.UP]:
            act_dict['object0:force'] = 1.0
        if curr_keys[KEY.DOWN]:
            act_dict['object0:force'] = -1.0
        if curr_keys[KEY.LEFT]:
            act_dict['object0:theta'] = 0.5
        if curr_keys[KEY.RIGHT]:
            act_dict['object0:theta'] = -0.5
        if curr_keys[KEY.SPACE] and not past_keys[KEY.SPACE]:
            paused = not paused
        if curr_keys[KEY.P] and not past_keys[KEY.P]:
            plotting = not plotting
        if curr_keys[KEY._1] and not past_keys[KEY._1]:
            dor = not dor

        if curr_keys[KEY.S]:
            delay *= 2.0 
        if curr_keys[KEY.F]:
            delay *= 0.5
        time.sleep(delay)

        if curr_keys[KEY.NUM_4]:
            pass
            # TODO: add support for rendering past images in traj

        if curr_keys[KEY.ESCAPE]: 
            exit()

        if not paused or (curr_keys[KEY.NUM_6] and not past_keys[KEY.NUM_6]):
            
            obs, rew, done, info = env.step(env.action_space.sample())
            #obs, rew, done, info = env.step(env.get_act_vec(act_dict))
            if done and dor:
                obs = env.reset()
            # print only the obs data that comes from object0
            #print(rew, utils.filter(env.get_obs_dict(obs, map=False), 'object0'))
            print(obs)
        img = env.render()
        if plotting:
            plt.imshow(img); plt.show()
        past_keys = {key: val for key, val in curr_keys.items()}