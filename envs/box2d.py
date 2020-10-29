import time
import itertools
from collections import defaultdict
from typing import NamedTuple, List, Set, Tuple, Dict

import copy
import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import utils
from envs.index_env import IndexEnv
A = utils.A

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
HULL_POLY = A[(-30,+0), (-20,+16), (+20,+16), (+30,+0), (+20,-16), (-20, -16) ]
hull_shape = polygonShape(vertices=[(x/SCALE,y/SCALE) for x,y in HULL_POLY])
VERT = 8/SCALE
SIDE = 20/SCALE

LEG_W, LEG_H = 8/SCALE, 20/SCALE

TABLE_H = 34 / SCALE
ARM_W, ARM_H = 6/SCALE, 26/SCALE
ARM_UP = 8/SCALE
OBJECT_H = 20/SCALE
FINGER_W, FINGER_H = 4/SCALE, 16/SCALE

SHAPES = {}
SHAPES['root'] = polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ])
SHAPES['lshoulder'] = SHAPES['lelbow'] = SHAPES['rshoulder'] = SHAPES['relbow'] = polygonShape(box = (ARM_W/2, ARM_H/2))
SHAPES['lfinger'] = SHAPES['rfinger'] = polygonShape(box = (FINGER_W/2, FINGER_H/2))
SHAPES['lhip'] = SHAPES['rhip'] = polygonShape(box=(LEG_W/2, LEG_H/2))
SHAPES['lknee'] = SHAPES['rknee'] = polygonShape(box=(0.8*LEG_W/2, LEG_H/2))

robot_bounds = {}
robot_bounds['lhip:angle'] = [-0.8, 1.1]
robot_bounds['lknee:angle'] = [-1.6, -0.1]
robot_bounds['rhip:angle'] = [-0.8, 1.1]
robot_bounds['rknee:angle'] = [-1.6, -0.1]
# arm
robot_bounds['shoulder:angle'] = [-2.57, 2.57]
#robot_bounds['elbow:angle'] = [-1.57, 1.57]
robot_bounds['elbow:angle'] = [-2.57, 2.57]  # toggle
# finger
robot_bounds['lfinger:angle'] = [-1.0, 2.5]
robot_bounds['rfinger:angle'] = [-2.5, 1.0]

class Body(NamedTuple):
    shape: polygonShape

class Joint(NamedTuple):
    parent: str
    angle: float
    anchorA: list
    anchorB: list
    limits: List[float]

class Agent(NamedTuple):
    name: str
    root_body: Body = Body(hull_shape)
    bodies: Dict[str, Body] = {
        'lhip': Body(SHAPES['lhip']),
        'lknee': Body(SHAPES['lknee']),
        'rhip': Body(SHAPES['lhip']),
        'rknee': Body(SHAPES['rknee']),
        'lshoulder': Body(SHAPES['lshoulder']),
        'lelbow': Body(SHAPES['lelbow']),
        'rshoulder': Body(SHAPES['rshoulder']),
        'relbow': Body(SHAPES['relbow']),
        }
    joints: Dict[str, Joint] = {
        'lhip': Joint('root', -0.5, (-SIDE, -VERT), (0, LEG_H/2), [-1.0, 0.5]),
        'lknee': Joint('lhip', -0.5, (0, -LEG_H/2), (0, LEG_H/2), [0.0, 0.5]),
        'rhip': Joint('root', 0.5, (SIDE, -VERT), (0, LEG_H/2), [-0.5, 0.5]),
        'rknee': Joint('rhip', -0.5, (0, -LEG_H/2), (0, LEG_H/2), [0.0, 0.5]),
        'lshoulder': Joint('root', 0.0, (-SIDE, VERT), (0, -ARM_H/2), [-1.0, 1.0]),
        'lelbow': Joint('lshoulder', -0.5, (0, ARM_H/2), (0, -ARM_H/2), [0.0, 0.5]),
        'rshoulder': Joint('root', 0.0, (SIDE, VERT), (0, -ARM_H/2), [-1.0, 1.0]),
        'relbow': Joint('rshoulder', -0.5, (0, ARM_H/2), (0, -ARM_H/2), [0.0, 0.5]),
        }

class Object(NamedTuple):
    name: str

class World(NamedTuple):
    agents: List[Agent] = []
    objects: List[Object] = [] 


MOTORS_TORQUE = defaultdict(lambda: 160)
#MOTORS_TORQUE['finger'] = 20
SPEEDS = defaultdict(lambda: 4)
SPEEDS['knee'] = 6


class B2D(IndexEnv):
    SCALE = SCALE
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    def __init__(self, w, cfg):
        EzPickle.__init__(self)
        self.w = w
        self.cfg = cfg
        self.VIEWPORT_W = cfg.env_size
        self.VIEWPORT_H = cfg.env_size
        self.world = Box2D.b2World(gravity=(0, -9.81))
        self.seed()
        self.viewer = None
        self.statics = {}
        self.dynbodies = {}
        # OBSERVATION + ACTION
        self.obs_info = {}
        self.act_info = {}
        for obj in self.w.objects:
            # TODO: add range options
            self.obs_info[f'{obj.name}:x:p'] = A[0, self.W]
            self.obs_info[f'{obj.name}:y:p'] = A[0, self.H]
            self.obs_info[f'{obj.name}:cos'] = A[-1, 1]
            self.obs_info[f'{obj.name}:sin'] = A[-1, 1]
        for agent in self.w.agents:
            self.obs_info[f'{agent.name}:root:x:p'] = A[0, self.W]
            self.obs_info[f'{agent.name}:root:y:p'] = A[0, self.H]
            self.obs_info[f'{agent.name}:root:cos'] = A[-1, 1]
            self.obs_info[f'{agent.name}:root:sin'] = A[-1, 1]
            for joint_name, joint in agent.joints.items():
                self.obs_info[f'{agent.name}:{joint_name}:theta'] = A[joint.limits]
                self.act_info[f'{agent.name}:{joint_name}:force'] = A[-1,1]
        if len(self.w.agents) == 0:
            self.act_info['dummy'] = A[-1, 1]  
        self.pack_info()

    @property
    def W(self):
        return self.VIEWPORT_W/B2D.SCALE
    @property
    def H(self):
        return self.VIEWPORT_H/B2D.SCALE

    def _destroy(self):
        for name, body in {**self.statics, **self.dynbodies}.items():
            self.world.DestroyBody(body)
        self.statics = {}
        self.dynbodies = {}

    def _reset_bodies(self, obs=None):
        if obs is not None: obs = utils.DWrap(obs, self.obs_info)
        self.dynbodies = {}
        self.joints = {}
        color = (0.9,0.4,0.4), (0.5,0.3,0.5)
        #color = (0.5,0.4,0.9), (0.3,0.3,0.5)
        # bodies
        def sample(name, x):
            return self.sample(f'{name}{x}') if obs is None else obs[f'{name}{x}']

        box_size = self.VIEWPORT_W / (B2D.SCALE*14.2222)
        for obj in self.w.objects:
            #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
            fixture = fixtureDef(shape=polygonShape(box=(box_size, box_size)), density=1)
            body = self.world.CreateDynamicBody(
                position=(sample(obj.name, ':x:p'), sample(obj.name, ':y:p')),
                angle=utils.get_angle(sample(obj.name, ':cos'), sample(obj.name, ':sin')),
                fixtures=fixture)
            body.color1, body.color2 = color
            self.dynbodies[obj.name] = body

        for agent in self.w.agents:
            # TODO: maybe create root first, and then add each of the body-joint pairs onto that.
            # this way we know where they should go. but first, let's get food.
            # like root body, then everything else is a body and a joint. joint specifies how the body attachs.
            fixture = fixtureDef(shape=agent.root_body.shape, density=1)
            name = agent.name+':root'
            root_xy = sample(name, ':x:p'), sample(name, ':y:p')
            root_xy = A[self.W//2,2]
            root_angle = utils.get_angle(sample(name, ':cos'), sample(name, ':sin'))
            root_angle = 0

            dyn = self.world.CreateDynamicBody(
                position=root_xy,
                angle=root_angle,
                fixtures=fixture)
            dyn.color1, dyn.color2 = color
            self.dynbodies[name] = dyn

            for bj_name in agent.joints:
                name = agent.name + ':' + bj_name
                body = agent.bodies[bj_name]
                joint = agent.joints[bj_name]
                parent_name = agent.name+':'+joint.parent

                fixture = fixtureDef(shape=body.shape, density=1)

                dyn = self.world.CreateDynamicBody(
                    position=self.dynbodies[parent_name].position-A[joint.anchorB],
                    #position=root_xy-A[joint.anchorA]-A[joint.anchorB],
                    angle=joint.angle,
                    fixtures=fixture)
                dyn.color1, dyn.color2 = color
                self.dynbodies[name] = dyn

                rjd = revoluteJointDef(
                    bodyA=self.dynbodies[parent_name],
                    bodyB=self.dynbodies[name],
                    localAnchorA=joint.anchorA,
                    localAnchorB=joint.anchorB,
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=0,
                    motorSpeed=0,
                    lowerAngle=joint.limits[0],
                    upperAngle=joint.limits[1],
                    )
                self.joints[name] = self.world.CreateJoint(rjd)

    def reset(self):
        self.ep_t = 0
        self._destroy()
        self.statics = {}
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.W, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.H)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.W, 0), (self.W, self.H)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.H), (self.W, self.H)]))
        self._reset_bodies()
        return self._get_obs().arr

    def _get_obs(self):
        obs = utils.DWrap(np.zeros(self.obs_size), self.obs_info, map=True)
        for obj in self.w.objects:
            body = self.dynbodies[obj.name]
            obs[f'{obj.name}:x:p'], obs[f'{obj.name}:y:p'] = body.position
            obs[f'{obj.name}:sin'] = np.sin(body.angle)
            obs[f'{obj.name}:cos'] = np.cos(body.angle)
            #obs[f'object{i}:x:v'], obs[f'object{i}:y:v'] = body.linearVelocity
        return obs

    def step(self, vec_action):
        self.ep_t += 1
        action = self.get_act_dict(vec_action)
        # TORQUE CONTROL
        for name in action:
            key = name.split(':force')[0]
            torque = MOTORS_TORQUE['default']
            speed = SPEEDS['default']
            self.joints[key].motorSpeed = float(speed * np.sign(action[name]))
            self.joints[key].maxMotorTorque = float(torque * np.clip(np.abs(action[name]), 0, 1))
        # RUN SIM STEP
        self.world.Step(1.0/FPS, 6*30, 2*30)
        obs = self._get_obs()
        reward = 0.0
        done = self.ep_t >= self.cfg.ep_len
        return obs.arr, reward, done, {}

    def visualize_obs(self, obs):
        self._reset_bodies(obs=obs)
        return self.render()

    def render(self, mode='rgb_array'):
        from envs import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W/B2D.SCALE, 0, self.VIEWPORT_H/B2D.SCALE)

        for name in self.dynbodies:
            body = self.dynbodies[name]
            for f in body.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=body.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=body.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=body.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=body.color2, linewidth=self.VIEWPORT_H/100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
