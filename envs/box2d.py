import pyglet
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
HULL_POLY = A[(-25,+0), (-20,+16), (+20,+16), (+25,+0), (+20,-16), (-20, -16) ]

class Body(NamedTuple):
    shape: polygonShape
class Joint(NamedTuple):
    parent: str
    angle: float
    anchorA: list
    anchorB: list
    limits: List[float]
# TODO: add drawing options and such
# TODO: add collision options, like different masks
class Agent(NamedTuple):
    name: str
    root_body: Body = None
    bodies: Dict[str, Body] = None
    joints: Dict[str, Joint] = None

def make_crab(name):
    VERT = 10/SCALE
    SIDE = 20/SCALE
    LEG_W, LEG_H = 8/SCALE, 16/SCALE
    ARM_W, ARM_H = 8/SCALE, 20/SCALE
    CLAW_W, CLAW_H = 4/SCALE, 16/SCALE
    SHAPES = {}
    SHAPES['root'] = polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ])
    SHAPES['arm'] = polygonShape(box = (ARM_W/2, ARM_H/2))
    SHAPES['hip'] = polygonShape(box=(LEG_W/2, LEG_H/2))
    SHAPES['knee'] = polygonShape(box=(0.8*LEG_W/2, LEG_H/2))
    SHAPES['claw'] = polygonShape(box = (CLAW_W/2, CLAW_H/2))
    return Agent(
        name=name,
        root_body=Body(SHAPES['root']),
        bodies = {
            'lhip': Body(SHAPES['hip']),
            'lknee': Body(SHAPES['knee']),
            'rhip': Body(SHAPES['hip']),
            'rknee': Body(SHAPES['knee']),
            'lshoulder': Body(SHAPES['arm']),
            'lelbow': Body(SHAPES['arm']),
            'rshoulder': Body(SHAPES['arm']),
            'relbow': Body(SHAPES['arm']),
            # left claw
            'llclaw0': Body(SHAPES['claw']),
            'llclaw1': Body(SHAPES['claw']),
            'lrclaw0': Body(SHAPES['claw']),
            'lrclaw1': Body(SHAPES['claw']),
            # right claw
            'rlclaw0': Body(SHAPES['claw']),
            'rlclaw1': Body(SHAPES['claw']),
            'rrclaw0': Body(SHAPES['claw']),
            'rrclaw1': Body(SHAPES['claw']),
            },
        joints = {
            'lhip': Joint('root', -0.5, (-SIDE, -VERT), (0, LEG_H/2), [-1.0, 0.5]),
            'lknee': Joint('lhip', 0.5, (0, -LEG_H/2), (0, LEG_H/2), [-0.5, 0.5]),
            'rhip': Joint('root', 0.5, (SIDE, -VERT), (0, LEG_H/2), [0.5, 1.0]),
            'rknee': Joint('rhip', -0.5, (0, -LEG_H/2), (0, LEG_H/2), [-0.5, 0.5]),
            'lshoulder': Joint('root', 1.0, (-SIDE, VERT), (0, -ARM_H/2), [-1.0, 2.0]),
            'lelbow': Joint('lshoulder', -0.5, (0, ARM_H/2), (0, -ARM_H/2), [-1.0, 2.0]),
            'rshoulder': Joint('root', -1.0, (SIDE, VERT), (0, -ARM_H/2), [-2.0, 1.0]),
            'relbow': Joint('rshoulder', 0.5, (0, ARM_H/2), (0, -ARM_H/2), [-2.0, 1.0]),
            # left claw
            'llclaw0': Joint('lelbow', 1.0, (0, ARM_H/2), (0, -CLAW_H/2), [-1.0, 2.0]),
            'llclaw1': Joint('llclaw0', -0.5, (0, CLAW_H/2), (0, -CLAW_H/2), [-0.1, 0.1]),
            'lrclaw0': Joint('lelbow', -1.0, (0, ARM_H/2), (0, -CLAW_H/2), [-2.0, 1.0]),
            'lrclaw1': Joint('lrclaw0', 0.5, (0, CLAW_H/2), (0, -CLAW_H/2), [-0.1, 0.1]),
            # right claw
            'rlclaw0': Joint('relbow', 1.0, (0, ARM_H/2), (0, -CLAW_H/2), [-1.0, 2.0]),
            'rlclaw1': Joint('rlclaw0', -0.5, (0, CLAW_H/2), (0, -CLAW_H/2), [-0.1, 0.1]),
            'rrclaw0': Joint('relbow', -1.0, (0, ARM_H/2), (0, -CLAW_H/2), [-2.0, 1.0]),
            'rrclaw1': Joint('rrclaw0', 0.5, (0, CLAW_H/2), (0, -CLAW_H/2), [-0.1, 0.1]),
            },)

class Object(NamedTuple):
    name: str

class World(NamedTuple):
    agents: List[Agent] = []
    objects: List[Object] = [] 

MOTORS_TORQUE = defaultdict(lambda: 160)
#MOTORS_TORQUE['claw'] = 20
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
        #self.VIEWPORT_W = 2*cfg.env_size
        self.VIEWPORT_H = cfg.env_size
        self.scale = 320/self.VIEWPORT_H

        #self.world = Box2D.b2World(gravity=(0, 0))
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
        for i in range(len(self.w.agents)):
            self.w.agents[i] = agent = make_crab(self.w.agents[i].name)
            self.obs_info[f'{agent.name}:root:x:p'] = A[0, self.W]
            self.obs_info[f'{agent.name}:root:y:p'] = A[0, self.H]
            self.obs_info[f'{agent.name}:root:cos'] = A[-1, 1]
            self.obs_info[f'{agent.name}:root:sin'] = A[-1, 1]
            for joint_name, joint in agent.joints.items():
                #self.obs_info[f'{agent.name}:{joint_name}:theta'] = A[joint.limits]
                self.obs_info[f'{agent.name}:{joint_name}:x:p'] = A[-0.5, 0.5]
                self.obs_info[f'{agent.name}:{joint_name}:y:p'] = A[-0.5, 0.5]
                self.obs_info[f'{agent.name}:{joint_name}:cos'] = A[-1, 1]
                self.obs_info[f'{agent.name}:{joint_name}:sin'] = A[-1, 1]
                # act
                self.act_info[f'{agent.name}:{joint_name}:force'] = A[-1,1]
        if len(self.w.agents) == 0: # because having a zero shaped array makes things break
            self.act_info['dummy'] = A[-1, 1] 
        self.pack_info()

    @property
    def W(self):
        return 10
    @property
    def H(self):
        return 10

    def _destroy(self):
        for name, body in {**self.statics, **self.dynbodies}.items():
            self.world.DestroyBody(body)
        self.statics = {}
        self.dynbodies = {}

    def _reset_bodies(self, inject_obs=None):
        if inject_obs is not None: inject_obs = utils.DWrap(inject_obs, self.obs_info)
        self.dynbodies = {}
        self.joints = {}
        # bodies
        def sample(namex, lr=-1.0, ur=None):
            if ur is None:
                ur = -lr
            if inject_obs is None:
                return utils.mapto(self.np_random.uniform(lr,ur), self.obs_info[f'{namex}'])
            else:
                return inject_obs[f'{namex}']

        box_size = self.H / 30.00
        for obj in self.w.objects:
            color = (0.5,0.4,0.9), (0.3,0.3,0.5)
            #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
            fixture = fixtureDef(shape=polygonShape(box=(box_size, box_size)), density=1.0, friction=1.0)
            body = self.world.CreateDynamicBody(
                position=(sample(obj.name+':x:p', -0.95), sample(obj.name+':y:p', -0.85, -0.80)),
                angle=np.arctan2(sample(obj.name+':sin'), sample(obj.name+':cos')),
                fixtures=fixture)
            body.color1, body.color2 = color
            self.dynbodies[obj.name] = body

        for agent in self.w.agents:
            color = (0.9,0.4,0.4), (0.5,0.3,0.5)
            # TODO: maybe create root first, and then add each of the body-joint pairs onto that.
            # this way we know where they should go. but first, let's get food.
            # like root body, then everything else is a body and a joint. joint specifies how the body attachs.
            fixture = fixtureDef(shape=agent.root_body.shape, density=1.0, categoryBits=0x0020, maskBits=0x001)
            name = agent.name+':root'
            #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.85, 0.80)
            root_xy = sample(name+':x:p', -0.7), sample(name+':y:p', -0.75, -0.70)
            #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.75, -0.70)
            root_angle = np.arctan2(sample(name+':sin'), sample(name+':cos'))
            if inject_obs is None: root_angle = 0
            dyn = self.world.CreateDynamicBody(
                position=root_xy,
                angle=root_angle,
                fixtures=fixture)
            dyn.color1, dyn.color2 = color
            self.dynbodies[name] = dyn

            parent_angles = {}
            parent_angles[name] = root_angle
            inject_angles = {}
            inject_angles[name] = root_angle

            for bj_name in agent.joints:
                name = agent.name + ':' + bj_name
                body = agent.bodies[bj_name]
                joint = agent.joints[bj_name]
                parent_name = agent.name+':'+joint.parent
                mangle = root_angle+joint.angle
                mangle = np.arctan2(np.sin(mangle), np.cos(mangle))
                parent_angles[name] = mangle
                fixture = fixtureDef(shape=body.shape, density=1, categoryBits=0x0020, maskBits=0x001)

                # parent rot
                aa_delta = A[joint.anchorA]
                pangle = parent_angles[parent_name]
                rot = utils.make_rot(pangle)
                aa_delta = rot.dot(aa_delta)
                # kid rot
                ab_delta = A[joint.anchorB]
                rot = utils.make_rot(mangle)
                ab_delta = rot.dot(ab_delta)
                dyn = self.world.CreateDynamicBody(
                    position=self.dynbodies[parent_name].position+aa_delta-ab_delta,
                    angle=mangle,
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
                self.joints[name] = jnt = self.world.CreateJoint(rjd)
                #self.joints[name].bodyB.transform.angle = sample(name+':theta')
                if inject_obs is not None:
                    jnt.bodyB.transform.position = A[root_xy] + A[(inject_obs[name+':x:p'], inject_obs[name+':y:p'])]
                    jnt.bodyB.transform.angle = np.arctan2(inject_obs[name+':sin'], inject_obs[name+':cos'])

    def reset(self):
        self.ep_t = 0
        self._destroy()
        self.statics = {}
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.W, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.H)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.W, 0), (self.W, self.H)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.H), (self.W, self.H)]))
        self._reset_bodies()
        #self.start_positions = {key: A[self.dynbodies[key].position] for key in self.dynbodies}
        return self._get_obs().arr

    def _get_obs(self):
        obs = utils.DWrap(np.zeros(self.obs_size), self.obs_info, map=True)
        for obj in self.w.objects:
            body = self.dynbodies[obj.name]
            obs[f'{obj.name}:x:p'], obs[f'{obj.name}:y:p'] = body.position
            obs[f'{obj.name}:cos'] = np.cos(body.angle)
            obs[f'{obj.name}:sin'] = np.sin(body.angle)
        for agent in self.w.agents:
            body = self.dynbodies[agent.name+':root']
            obs[f'{agent.name}:root:x:p'], obs[f'{agent.name}:root:y:p'] = root_xy = body.position
            obs[f'{agent.name}:root:cos'] = np.cos(body.angle)
            obs[f'{agent.name}:root:sin'] = np.sin(body.angle)
            for joint_name, joint in agent.joints.items():
                jnt = self.joints[f'{agent.name}:{joint_name}']
                obs[f'{agent.name}:{joint_name}:x:p'], obs[f'{agent.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position - root_xy
                obs[f'{agent.name}:{joint_name}:cos'] = np.cos(jnt.bodyB.transform.angle)
                obs[f'{agent.name}:{joint_name}:sin'] = np.sin(jnt.bodyB.transform.angle)
                #obs[f'{agent.name}:{joint_name}:theta'] = jnt.angle
        return obs

    def step(self, vec_action):
        self.ep_t += 1
        action = self.get_act_dict(vec_action)
        # TORQUE CONTROL
        for name in action:
            if name == 'dummy':
                continue
            key = name.split(':force')[0]
            torque = MOTORS_TORQUE['default']
            speed = SPEEDS['default']
            self.joints[key].motorSpeed = float(speed * np.sign(action[name]))
            self.joints[key].maxMotorTorque = float(torque * np.clip(np.abs(action[name]), 0, 1))
        #self.end_positions = {key: A[self.dynbodies[key].position] for key in self.dynbodies}
        # RUN SIM STEP
        self.world.Step(1.0/FPS, 6*30, 2*30)
        obs = self._get_obs()
        reward = 0.0
        done = self.ep_t >= self.cfg.ep_len
        return obs.arr, reward, done, {}

    def visualize_obs(self, obs, text=''):
        self._reset_bodies(inject_obs=obs)
        return self.render(text=text)

    def render(self, mode='rgb_array', text=''):
        from envs import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.scale*self.VIEWPORT_W/(SCALE), 0, self.scale*self.VIEWPORT_H/(SCALE))

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
                    self.viewer.draw_polygon(A[path], color=body.color1)
                    #self.viewer.draw_polygon(A[path]/self.scale, color=body.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(A[path], color=body.color2, linewidth=self.VIEWPORT_H/100)
                    #self.viewer.draw_polyline(A[path]/self.scale, color=body.color2, linewidth=self.VIEWPORT_H/100)

            if 'root' in name:
                rot = utils.make_rot(body.angle)
                pos = body.position
                X = 0.1
                X2 = 0.2
                t = rendering.Transform(translation=pos+rot.dot(A[X2, +X]))
                self.viewer.draw_circle(X, 20, color=body.color2).add_attr(t)
                #self.viewer.draw_circle(X, 20, color=(0.8, 0.8, 0.8), filled=False, linewidth=5).add_attr(t)
                t = rendering.Transform(translation=pos+rot.dot(A[-X2, +X]))
                self.viewer.draw_circle(X, 20, color=body.color2).add_attr(t)
               # self.viewer.draw_circle(X, 20, color=(0.8, 0.8, 0.8), filled=False, linewidth=5).add_attr(t)
                t = rendering.Transform(translation=pos+rot.dot(A[0.0, -X2]), rotation=body.angle)
                self.viewer.draw_line((0, 0), (X2, X), color=body.color2).add_attr(t)
                self.viewer.draw_line((0, 0), (-X2, X), color=body.color2).add_attr(t)

        return self.viewer.render(return_rgb_array = mode=='rgb_array', text=text, size=4//self.scale)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None