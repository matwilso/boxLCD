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
VIEWPORT_W = 200
VIEWPORT_H = 200
W = VIEWPORT_W/SCALE
H = VIEWPORT_H/SCALE

POLYS = {}
POLYS['fish0'] =[ (-5, 10), (-10,+5), (-10,-5), (0,-15), (+10,-5), (+10,+5), (5, 10)]
POLYS['fish1'] =[ (-5, 10), (-10,+5), (-10,-5), (0,-15), (+10,-5), (+10,+5), (5, 10)]
BLOCK_OFFSET = 10/SCALE
FISH_OFFSET = 22/SCALE

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
    def EndContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]

class Fishes(BaseIndexEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self):
        EzPickle.__init__(self)
        self.fish_list = ['fish0', 'fish1']
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World()#)gravity=(0,0))
        self.dynbodies = None
        self.statics = None
        self.prev_reward = None
        self.ep_t = 0
        # the actual true limits of our environment
        self.max_linear = 2.0
        self.max_angular = 2.0
        self.threshold = 1
        self.timeout = 500
        # OBSERVATION
        self.obs_info = {}
        # fish0
        for fish in self.fish_list:
            self.obs_info[f'{fish}:x'] = A[FISH_OFFSET, W-FISH_OFFSET]
            self.obs_info[f'{fish}:y'] = A[FISH_OFFSET, H-FISH_OFFSET]
            self.obs_info[f'{fish}:gx'] = A[FISH_OFFSET, W-FISH_OFFSET]
            self.obs_info[f'{fish}:gy'] = A[FISH_OFFSET, H-FISH_OFFSET]
            self.obs_info[f'{fish}:cos'] = A[-1, 1]
            self.obs_info[f'{fish}:sin'] = A[-1, 1]
            # extra we can add later if needed
            #self.obs_info[f'{fish}:vel'] = [0, self.max_linear]
            #self.obs_info[f'{fish}:thetavel'] = [-self.max_angular, self.max_angular]
        # ACTION 
        self.act_info = {}
        for fish in self.fish_list:
            self.act_info[f'{fish}:force'] = A[-1, 1]  
            self.act_info[f'{fish}:theta'] = A[-1, 1]
        self.pack_info()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if self.dynbodies is None:
            return
        self.world.contactListener = None
        # TODO: turn all the bodies in a dictionary of bodies
        for name, body in self.statics.items():
            self.world.DestroyBody(body)
        self.statics = None
        for name, body in self.dynbodies.items():
            self.world.DestroyBody(body)
        self.dynbodies = None

    def sample(self, name):
        return self.np_random.uniform(*self.obs_info[name])

    def reset(self):
        self.stasis_count = 0
        self.prev_reward = None
        self.ep_t = 0
        self.last_state = np.inf
        self.stasis_count = 0
        self.done = False
        
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.statics = {}
        self.statics['wall1'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.statics['wall2'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (0, H)]) )
        self.statics['wall3'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(W, 0), (W, H)]) )
        self.statics['wall4'] = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, H), (W, H)]) )
        self.dynbodies = {}
        colors = {}
        colors['fish0'] = (0.9,0.4,0.4), (0.5,0.3,0.5)
        colors['fish1'] = (0.5,0.4,0.9), (0.3,0.3,0.5)
        # bodies
        for key in self.fish_list:
            sample = lambda x: self.sample(f'{key}{x}')
            fixture = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x,y in POLYS[key]]),
                density=5.0,
                friction=0.9,
                categoryBits=0x001,
                maskBits=0x001,)
            body = self.world.CreateDynamicBody(
                position=(sample(':x'), sample(':y')),
                angle=utils.get_angle(sample(':cos'), sample(':sin')),
                angularDamping=9,
                linearDamping=5,
                fixtures=fixture)
            body.color1, body.color2 = colors[key]
            self.dynbodies[key] = body
        self.dynbodies = utils.sortdict(self.dynbodies)

        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.goal = {}
        for fish in self.fish_list:
            for xy in 'xy':
                name = f'{fish}:g{xy}'
                self.goal[name] = self.sample(name)
        return self._get_obs().arr

    def _get_obs(self):
        obs = utils.DWrap(np.zeros(self.obs_size), self.obs_info, map=True)
        for key in self.fish_list:
            body = self.dynbodies[key]
            obs[f'{key}:x'], obs[f'{key}:y'] = body.position
            obs[f'{key}:sin'] = np.sin(body.angle)
            obs[f'{key}:cos'] = np.cos(body.angle)
            #obs['fish0:vel'] = np.linalg.norm(fish0.linearVelocity)
            #obs['fish0:thetavel'] = fish0.angularVelocity
        for key in self.goal: obs[key] = self.goal[key]
        return obs

    def step(self, vec_action):
        self.ep_t += 1
        action = self.get_act_dict(vec_action)
        # APPLY FORCE AND TORQUE
        for key in self.fish_list:
            body = self.dynbodies[key]
            mag = 120*action[f'{key}:force']
            if mag < 0: mag *= 0.25  # you can only go half-speed in reverse, to incentivize forward movement
            theta = 40*action[f'{key}:theta']
            fx, fy = -mag*np.sin(body.angle), mag*np.cos(body.angle)
            body_pos = body.position 
            point = body_pos# + body.localCenter
            force = (fx, fy)
            body.ApplyForce(force=force, point=point, wake=True)
            body.ApplyTorque(theta, wake=True)
        # RUN SIM STEP
        self.world.Step(1.0/FPS, 6*30, 2*30)
        # CLIP SPEEDS
        for key in self.fish_list:
            body = self.dynbodies[key]
            # clip linear and angular
            if np.linalg.norm(body.linearVelocity) > self.max_linear:
                body.linearVelocity *= (self.max_linear / np.linalg.norm(body.linearVelocity)) 
            body.angularVelocity = np.clip(body.angularVelocity, -self.max_angular, self.max_angular)

        obs = self._get_obs()
        # REWARD
        state = []
        goal = []
        for fish in self.fish_list:
            for xy in 'xy':
                state += [f'{fish}:{xy}']
                goal += [f'{fish}:g{xy}']
        #state = ['fish0:x', 'fish0:y']#, 'fish1:x', 'fish1:y']
        #goal = ['fish0:gx', 'fish0:gy']#, 'fish1:gx', 'fish1:gy']
        delta = np.linalg.norm(obs[state] - obs[goal])
        reward = -delta/10.0
        # DONE AND RETURN 
        self.info = {}
        if delta < self.threshold:
            self.done = True
            self.info['done_for'] = 'done_success'
        if self.ep_t >= self.timeout:
            self.done = True
            self.info['done_for'] = 'done_timeout'
        return obs.arr, reward, self.done, self.info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for key in reversed(list(self.dynbodies.keys())):
            obj = self.dynbodies[key]
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                if self.done and 'done_for' in self.info and self.info['done_for'] == 'done_success':
                    color1 = (0.1, 0.8, 0.1)
                    color2 = (0.7, 0.9, 0.7)
                elif self.done and 'done_for' in self.info and self.info['done_for'] == 'done_timeout':
                    color1 = (0.1, 0.1, 0.1)
                    color2 = (0.9, 0.9, 0.9)
                else:
                    color1 = obj.color1
                    color2 = obj.color2
                self.viewer.draw_polygon(path, color=color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=color2, linewidth=2)

            if key in self.fish_list:
                pos = (self.goal[f'{key}:gx'], self.goal[f'{key}:gy'])
                t = rendering.Transform(translation=pos)
                radius = self.threshold
                #self.viewer.draw_circle(radius, 20, color=obj.color1).add_attr(t)
                #self.viewer.draw_circle(radius, 20, color=obj.color2, filled=True).add_attr(t)
                if self.done:
                    color = (0.7, 0.9, 0.7) if 'done_for' in self.info and self.info['done_for'] == 'done_success' else (0.1, 0.1, 0.1)
                else:
                    color = obj.color2
                self.viewer.draw_circle(radius, 20, color=color, filled=False, linewidth=1).add_attr(t)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    env = Fishes()
    env.reset()
    env.render()
    KEY = pyglet.window.key
    keys = KEY.KeyStateHandler()
    env.viewer.window.push_handlers(keys)
    window = env.viewer.window

    while True:
        action = env.action_space.sample()
        action = np.zeros_like(action)
        act_dict = env.get_act_dict(action)

        if keys[KEY._0]: 
            env.reset()
            time.sleep(0.1)
        if keys[KEY.UP]:
            act_dict['fish0:force'] = 1.0
        if keys[KEY.DOWN]:
            act_dict['fish0:force'] = -1.0
        if keys[KEY.LEFT]:
            act_dict['fish0:theta'] = 0.5
        if keys[KEY.RIGHT]:
            act_dict['fish0:theta'] = -0.5
        if keys[KEY.W]:
            act_dict['fish1:force'] = 1.0
        if keys[KEY.S]:
            act_dict['fish1:force'] = -1.0
        if keys[KEY.A]:
            act_dict['fish1:theta'] = 0.5
        if keys[KEY.D]:
            act_dict['fish1:theta'] = -0.5

        if keys[KEY.ESCAPE]: 
            exit()

        obs, rew, done, info = env.step(env.get_act_vec(act_dict))
        # print only the obs data that comes from fish0
        print(rew, utils.filter(env.get_obs_dict(obs, map=False), 'fish0'))
        env.render()