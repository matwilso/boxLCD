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
from envs.base import BaseIndexEnv
import pyglet
A = utils.A

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 64
VIEWPORT_H = 64
W = VIEWPORT_W/SCALE
H = VIEWPORT_H/SCALE

class Box(BaseIndexEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self):
        EzPickle.__init__(self)
        self.num_objects = 1
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, -9.81))
        self.dynbodies = {}
        self.statics = {}
        self.ep_t = 0
        # the actual true limits of our environment
        self.threshold = 1
        # OBSERVATION
        self.obs_info = {}
        for i in range(self.num_objects): 
            self.obs_info[f'object{i}:x:p'] = A[0, W]
            self.obs_info[f'object{i}:y:p'] = A[0, H]
            self.obs_info[f'object{i}:x:v'] = A[-10, 10]
            self.obs_info[f'object{i}:y:v'] = A[-10, 10]
            self.obs_info[f'object{i}:cos'] = A[-1, 1]
            self.obs_info[f'object{i}:sin'] = A[-1, 1]
            # extra we can add later if needed
            #self.obs_info[f'{object}:thetavel'] = [-self.max_angular, self.max_angular]
        # ACTION 
        self.act_info = {}
        self.act_info['dummy'] = A[-1, 1]  
        #for object in self.object_list:
        #    self.act_info[f'{object}:force'] = A[-1, 1]  
        #    self.act_info[f'{object}:theta'] = A[-1, 1]
        self.pack_info()

    def _destroy(self):
        for name, body in self.statics.items():
            self.world.DestroyBody(body)
        for name, body in self.dynbodies.items():
            self.world.DestroyBody(body)
        self.statics = {}
        self.dynamics = {}

    def reset(self):
        self.ep_t = 0
        self._destroy()
        self.statics = {}
        self.statics['wall1'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.statics['wall2'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (0, H)]) )
        self.statics['wall3'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(W, 0), (W, H)]) )
        self.statics['wall4'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, H), (W, H)]) )
        self.dynbodies = {}
        color = (0.9,0.4,0.4), (0.5,0.3,0.5)
        #color = (0.5,0.4,0.9), (0.3,0.3,0.5)
        # bodies
        for i in range(self.num_objects):
            sample = lambda x: self.sample(f'object{i}{x}')
            #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
            fixture = fixtureDef(shape=polygonShape(box=(0.15, 0.15)), density=1)
            body = self.world.CreateDynamicBody(
                position=(sample(':x:p'), sample(':y:p')),
                angle=utils.get_angle(sample(':cos'), sample(':sin')),
                fixtures=fixture)
            body.color1, body.color2 = color
            self.dynbodies[i] = body

        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        return self._get_obs().arr

    def _get_obs(self):
        obs = utils.DWrap(np.zeros(self.obs_size), self.obs_info, map=True)
        for i in range(self.num_objects):
            body = self.dynbodies[i]
            obs[f'object{i}:x:p'], obs[f'object{i}:y:p'] = body.position
            obs[f'object{i}:sin'] = np.sin(body.angle)
            obs[f'object{i}:cos'] = np.cos(body.angle)
            obs[f'object{i}:x:v'], obs[f'object{i}:y:v'] = body.linearVelocity
        return obs

    def step(self, vec_action):
        self.ep_t += 1
        #action = self.get_act_dict(vec_action)
        # APPLY FORCE AND TORQUE
        #for i in range(self.num_objects):
        #    body = self.dynbodies[i]
        #    mag = 120*action[f'{key}:force']
        #    if mag < 0: mag *= 0.25  # you can only go half-speed in reverse, to incentivize forward movement
        #    theta = 40*action[f'{key}:theta']
        #    fx, fy = -mag*np.sin(body.angle), mag*np.cos(body.angle)
        #    body_pos = body.position 
        #    point = body_pos# + body.localCenter
        #    force = (fx, fy)
        #    body.ApplyForce(force=force, point=point, wake=True)
        #    body.ApplyTorque(theta, wake=True)
        # RUN SIM STEP
        self.world.Step(1.0/FPS, 6*30, 2*30)
        # CLIP SPEEDS
        for i in range(self.num_objects):
            body = self.dynbodies[i]
            ## clip linear and angular
            #if np.linalg.norm(body.linearVelocity) > self.max_linear:
            #    body.linearVelocity *= (self.max_linear / np.linalg.norm(body.linearVelocity)) 
            #body.angularVelocity = np.clip(body.angularVelocity, -self.max_angular, self.max_angular)

        obs = self._get_obs()
        reward = 0.0
        done = self.ep_t >= 100
        return obs.arr, reward, done, {}

    def render(self, mode='rgb_array'):
        from envs import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for i in range(self.num_objects):
            obj = self.dynbodies[i]
            for f in obj.fixtures:
                trans = f.body.transform

                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=0.640)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = Box()
    env.reset()
    env.render()
    KEY = pyglet.window.key
    keys = KEY.KeyStateHandler()
    env.viewer.window.push_handlers(keys)
    window = env.viewer.window
    paused = False
    traj = []
    past_keys = {}

    while True:
        action = env.action_space.sample()
        action = np.zeros_like(action)
        act_dict = env.get_act_dict(action)
        curr_keys = defaultdict(lambda: False)
        curr_keys.update({key: val for key, val in keys.items()})

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

        if curr_keys[KEY.NUM_4]:
            pass
            # TODO: add support for rendering past images in traj

        if curr_keys[KEY.ESCAPE]: 
            exit()

        if not paused or (curr_keys[KEY.NUM_6] and not past_keys[KEY.NUM_6]):
            obs, rew, done, info = env.step(env.get_act_vec(act_dict))
            if done:
                obs = env.reset()
            # print only the obs data that comes from object0
            #print(rew, utils.filter(env.get_obs_dict(obs, map=False), 'object0'))
        obs = env.render()
        #plt.imshow(obs); plt.show()
        past_keys = {key: val for key, val in curr_keys.items()}