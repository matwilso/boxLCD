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
A = utils.A
from envs.index_env import IndexEnv
from envs.world_defs import FPS, SCALE, MAKERS

class B2D(IndexEnv):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : FPS
  }
  @property
  def WIDTH(self):
    return int(self.F.env_wh_ratio*10)
  @property
  def HEIGHT(self):
    return 10

  def __init__(self, w, F):
    """world, Hps"""
    EzPickle.__init__(self)
    self.w = w
    self.F = F
    self.SCALE = SCALE
    self.FPS = FPS
    self.scroll = 0.0
    self.VIEWPORT_H = self.F.env_size
    self.VIEWPORT_W = int(self.F.env_wh_ratio*self.F.env_size)
    self.scale = 320/self.VIEWPORT_H
    self.SPEEDS = defaultdict(lambda: 8) if self.F.use_speed else defaultdict(lambda: 6)
    self.MOTORS_TORQUE = defaultdict(lambda: 150) if float(self.F.env_version) < 0.3 or float(self.F.env_version) >= 0.6 else defaultdict(lambda: 100)
    if float(self.F.env_version) >= 0.6:
      self.MOTORS_TORQUE = defaultdict(lambda: 100)
      self.SPEEDS = defaultdict(lambda: 6)
      self.SPEEDS['hip'] = 10
      self.SPEEDS['knee'] = 10
      self.MOTORS_TORQUE['hip'] = 150
      self.MOTORS_TORQUE['knee'] = 150

    self.world = Box2D.b2World(gravity=self.w.gravity)
    self.seed()
    self.viewer = None
    self.statics = {}
    self.dynbodies = {}
    # OBSERVATION + ACTION
    self.obs_info = {}
    self.act_info = {}
    #self.obs_info['pad'] = [-1,1]
    for obj in self.w.objects:
      # TODO: add range options
      self.obs_info[f'{obj.name}:x:p'] = A[0, self.WIDTH]
      self.obs_info[f'{obj.name}:y:p'] = A[0, self.HEIGHT]
      self.obs_info[f'{obj.name}:cos'] = A[-1, 1]
      self.obs_info[f'{obj.name}:sin'] = A[-1, 1]

    for i in range(len(self.w.agents)):
      self.w.agents[i] = agent = MAKERS[self.F.cname](self.w.agents[i].name, SCALE, self.F)
      self.cname = self.F.cname+'0'
      self.obs_info[f'{agent.name}:root:x:p'] = A[0, self.WIDTH]
      self.obs_info[f'{agent.name}:root:y:p'] = A[0, self.HEIGHT]
      self.obs_info[f'{agent.name}:root:cos'] = A[-1, 1]
      self.obs_info[f'{agent.name}:root:sin'] = A[-1, 1]

      for joint_name, joint in agent.joints.items():
        if self.F.root_offset:
          self.obs_info[f'{agent.name}:{joint_name}:x:p'] = A[-2.0, 2.0]
          self.obs_info[f'{agent.name}:{joint_name}:y:p'] = A[-2.0, 2.0]
        else:
          self.obs_info[f'{agent.name}:{joint_name}:x:p'] = A[0, self.WIDTH]
          self.obs_info[f'{agent.name}:{joint_name}:y:p'] = A[0, self.HEIGHT]
        self.obs_info[f'{agent.name}:{joint_name}:cos'] = A[-1, 1]
        self.obs_info[f'{agent.name}:{joint_name}:sin'] = A[-1, 1]

        if joint.limits[0] != joint.limits[1]:
          # act
          if self.F.use_speed:
            self.act_info[f'{agent.name}:{joint_name}:speed'] = A[-1,1]
          else:
            self.act_info[f'{agent.name}:{joint_name}:force'] = A[-1,1]
            
    if len(self.w.agents) == 0: # because having a zero shaped array makes things break
      self.act_info['dummy'] = A[-1, 1] 
    self.pack_info()

  def _destroy(self):
    for name, body in {**self.statics, **self.dynbodies}.items():
      self.world.DestroyBody(body)
    self.statics = {}
    self.dynbodies = {}

  def _reset_bodies(self):
    self.dynbodies = {}
    self.joints = {}
    # bodies
    def sample(namex, lr=-1.0, ur=None):
      if ur is None: ur = -lr
      return utils.mapto(self.np_random.uniform(lr,ur), self.obs_info[f'{namex}'])

    for agent in self.w.agents:
      color = (0.9,0.4,0.4,1.0), (0.5,0.3,0.5,1.0)
      # TODO: maybe create root first, and then add each of the body-joint pairs onto that.
      # this way we know where they should go. but first, let's get food.
      # like root body, then everything else is a body and a joint. joint specifies how the body attachs.
      root_body = agent.root_body

      if float(self.F.env_version) >= 0.4:
        fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits, friction=1.0)
      else:
        fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits)

      name = agent.name+':root'
      if self.cname == 'crab0':
        if self.F.env_wh_ratio == 2:
          root_xy = A[sample(name+':x:p', -0.9, 0.95), sample(name+':y:p', -0.7, -0.7)]
        elif self.F.env_wh_ratio < 1:
          rat = self.F.env_wh_ratio
          root_xy = A[sample(name+':x:p', -0.7*rat, 0.8*rat), sample(name+':y:p', -0.7, -0.70)]
        else:
          root_xy = A[sample(name+':x:p', -0.7, 0.8), sample(name+':y:p', -0.7, -0.70)]
      elif self.cname == 'luxo0':
        root_xy = A[sample(name+':x:p', -0.7, 0.7), sample(name+':y:p', -0.6, -0.60)]
      else:
        root_xy = A[sample(name+':x:p', *agent.rangex), sample(name+':y:p', *agent.rangey)]

      #root_xy = A[sample(name+':x:p', -0.85, -0.8), sample(name+':y:p', -0.75, -0.70)]
      #root_xy = A[sample(name+':x:p', -0.85, -0.8), sample(name+':y:p', -0.50, -0.50)]
      #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.85, 0.80)
      #root_xy = A[sample(name+':x:p', -0.7), sample(name+':y:p', -0.75, -0.70)]
      #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.75, -0.70)
      def comp_angle(name, body, base_pos):
        cxy = (sample(f'{name}:cx:p'), sample(f'{name}:cy:p')) - A[base_pos]
        base_angle = np.arctan2(*body.shape.vertices[-1][::-1])
        offset_angle = np.arctan2(*cxy[::-1])
        ang = offset_angle - base_angle
        return np.arctan2(np.sin(ang), np.cos(ang))

      if self.F.all_corners:
        body = root_body
        root_angle = comp_angle(name, body, root_xy)
      else:
        root_angle = np.arctan2(sample(name+':sin'), sample(name+':cos'))
        
      if not agent.rand_angle:
        root_angle = 0
      dyn = self.world.CreateDynamicBody(
        position=root_xy,
        angle=root_angle,
        fixtures=fixture,
        angularDamping=agent.angularDamping,
        linearDamping=agent.linearDamping,
        )
      dyn.color1, dyn.color2 = color
      self.dynbodies[name] = dyn

      parent_angles = {}
      parent_angles[name] = root_angle

      for bj_name in agent.joints:
        name = agent.name + ':' + bj_name
        body = agent.bodies[bj_name]
        joint = agent.joints[bj_name]
        parent_name = agent.name+':'+joint.parent
        mangle = root_angle+joint.angle
        mangle = np.arctan2(np.sin(mangle), np.cos(mangle))
        parent_angles[name] = mangle

        fixture = fixtureDef(shape=body.shape, density=1, restitution=0.0, categoryBits=body.categoryBits, maskBits=body.maskBits, friction=body.friction)

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
          enableLimit=joint.limited,
          maxMotorTorque=self.MOTORS_TORQUE[name],
          motorSpeed=0,
          lowerAngle=joint.limits[0],
          upperAngle=joint.limits[1],
          )
        self.joints[name] = jnt = self.world.CreateJoint(rjd)
        #self.joints[name].bodyB.transform.angle = sample(name+':theta')

    for obj in self.w.objects:
      obj_size = obj.size*self.HEIGHT / 30.00
      color = (0.5,0.4,0.9,1.0), (0.3,0.3,0.5,1.0)
      #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
      obj_shapes = {'circle': circleShape(radius=obj_size, pos=(0,0)), 'box': (polygonShape(box=(obj_size, obj_size)))}
      shape = list(obj_shapes.keys())[np.random.randint(len(obj_shapes))] if obj.shape == 'random' else obj.shape
      shape = obj_shapes[shape]
      fixture = fixtureDef(shape=shape, density=obj.density, friction=obj.friction, categoryBits=obj.categoryBits)
      if len(self.w.agents) == 0:
        pos = A[(sample(obj.name+':x:p', -0.95, 0.95), sample(obj.name+':y:p', -0.95, 0.95))]
      else:
        pos = A[(sample(obj.name+':x:p', -0.95), sample(obj.name+':y:p', -0.9, -0.25))]

      if self.F.all_corners:
        samp = sample(obj.name+':cx:p'), sample(obj.name+':cy:p')
        angle = np.arctan2(*(pos - samp))
      else:
        angle = np.arctan2(sample(obj.name+':sin'), sample(obj.name+':cos'))

      body = self.world.CreateDynamicBody(
        position=pos,
        angle=angle,
        fixtures=fixture,
        angularDamping=0.1,
        #linearDamping=0.5,
        linearDamping=obj.damping,
        )
      body.color1, body.color2 = color
      self.dynbodies[obj.name] = body



  def reset(self, inject_obs=None):
    self.ep_t = 0
    self._destroy()
    self.statics = {}
    if self.F.walls:
      if float(self.F.env_version) <= 0.1:
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.WIDTH, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.HEIGHT)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.WIDTH, 0), (self.WIDTH, self.HEIGHT)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.HEIGHT), (self.WIDTH, self.HEIGHT)]))
      else:
        X = 0.6
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.WIDTH+X, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.HEIGHT+X)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.WIDTH+X, 0), (self.WIDTH+X, self.HEIGHT)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.HEIGHT+X), (self.WIDTH+X, self.HEIGHT+X)]))
    else:
        self.statics['floor'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(-1000*self.WIDTH, 0), (1000*self.WIDTH, 0)]))
    self._reset_bodies()
    #self.world.Step(0.001/FPS, 6*30, 2*30)
    if inject_obs is not None:
      def comp_angle(name, body, base_pos):
        cxy = (inject_obs[f'{name}:cx:p'], inject_obs[f'{name}:cy:p']) - A[base_pos]
        base_angle = np.arctan2(*body.shape.vertices[-1][::-1])
        offset_angle = np.arctan2(*cxy[::-1])
        ang = offset_angle - base_angle
        return np.arctan2(np.sin(ang), np.cos(ang))

      inject_obs = utils.DWrap(np.array(inject_obs).astype(np.float), self.obs_info)

      if len(self.w.agents) != 0:
        name = self.w.agents[0].name+':root'
        root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

      for obj in self.w.objects:
        name = obj.name
        self.dynbodies[name].position = inject_obs[f'{name}:x:p', f'{name}:y:p']

      for agent in self.w.agents:
        name = agent.name+':root'
        self.dynbodies[f'{name}'].position = root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

        if self.F.all_corners:
          self.dynbodies[f'{name}'].angle = root_angle = comp_angle(name, agent.root_body, root_xy)
        else:
          self.dynbodies[f'{name}'].angle = root_angle = np.arctan2(inject_obs(name+':sin'), inject_obs(name+':cos'))
        parent_angles = {}
        parent_angles[name] = root_angle

        for bj_name in agent.joints:
          name = agent.name + ':' + bj_name
          body = agent.bodies[bj_name]
          joint = agent.joints[bj_name]
          if (float(self.F.env_version) < 0.5 and float(self.F.env_version) >= 0.3) and joint.limits[0] == joint.limits[1]:
            continue
          parent_name = agent.name+':'+joint.parent
          mangle = root_angle+joint.angle
          mangle = np.arctan2(np.sin(mangle), np.cos(mangle))
          parent_angles[name] = mangle
          # parent rot
          aa_delta = A[joint.anchorA]
          pangle = parent_angles[parent_name]
          rot = utils.make_rot(pangle)
          aa_delta = rot.dot(aa_delta)
          # kid rot
          ab_delta = A[joint.anchorB]
          rot = utils.make_rot(mangle)
          ab_delta = rot.dot(ab_delta)
          if self.F.root_offset:
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[root_xy] + A[(inject_obs[name+':x:p'], inject_obs[name+':y:p'])]
          else:
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[(inject_obs[name+':x:p'], inject_obs[name+':y:p'])]

          if self.F.all_corners:
            offset_angle = comp_angle(name, agent.bodies[name.split(':')[1]], pos)
          else:
            offset_angle = np.arctan2(inject_obs[name+':sin'], inject_obs[name+':cos'])
            if self.F.angular_offset:
              offset_angle = root_angle + offset_angle
              offset_angle = np.arctan2(np.sin(offset_angle), np.cos(offset_angle))
          self.dynbodies[name].angle = offset_angle
    if not self.F.walls:
      self.scroll = self.dynbodies[f'{self.cname}:root'].position.x - self.VIEWPORT_W/SCALE/2
    return self._get_obs().arr

  def _get_obs(self):
    obs = utils.DWrap(np.zeros(self.obs_size), self.obs_info)
    for obj in self.w.objects:
      body = self.dynbodies[obj.name]
      obs[f'{obj.name}:x:p'], obs[f'{obj.name}:y:p'] = body.position
      if self.F.all_corners:
        obs[f'{obj.name}:cx:p', f'{obj.name}:cy:p'] = A[body.transform*body.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{obj.name}:cos'] = np.cos(body.angle)
        obs[f'{obj.name}:sin'] = np.sin(body.angle)

    for agent in self.w.agents:
      root = self.dynbodies[agent.name+':root']
      obs[f'{agent.name}:root:x:p'], obs[f'{agent.name}:root:y:p'] = root_xy = root.position

      if self.F.obj_offset:
        obs[f'{obj.name}:xd:p'], obs[f'{obj.name}:yd:p'] = obs[f'{obj.name}:x:p', f'{obj.name}:y:p'] - A[root_xy]

      if self.F.all_corners:
        obs[f'{agent.name}:root:cx:p', f'{agent.name}:root:cy:p'] = A[root.transform*root.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{agent.name}:root:cos'] = np.cos(root.angle)
        obs[f'{agent.name}:root:sin'] = np.sin(root.angle)
      for joint_name, joint in agent.joints.items():
        jnt = self.joints[f'{agent.name}:{joint_name}']
        if (float(self.F.env_version) < 0.5 and float(self.F.env_version) >= 0.3) and joint.limits[0] == joint.limits[1]:
          continue
        if self.F.compact_obs:
          obs[f'{agent.name}:{joint_name}:angle'] = jnt.angle
        else:
          if self.F.root_offset:
            obs[f'{agent.name}:{joint_name}:x:p'], obs[f'{agent.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position - root_xy
          else:
            obs[f'{agent.name}:{joint_name}:x:p'], obs[f'{agent.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position
          if self.F.angular_offset:
            angle = jnt.bodyB.transform.angle - root.angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
          else:
            angle = jnt.bodyB.transform.angle
          if self.F.all_corners:
            obs[f'{agent.name}:{joint_name}:cx:p', f'{agent.name}:{joint_name}:cy:p'] = A[jnt.bodyB.transform*jnt.bodyB.fixtures[0].shape.vertices[-1]]
          else:
            obs[f'{agent.name}:{joint_name}:cos'] = np.cos(angle)
            obs[f'{agent.name}:{joint_name}:sin'] = np.sin(angle)
    return obs

  def step(self, vec_action):
    self.ep_t += 1
    action = self.get_act_dict(vec_action)
    # TORQUE CONTROL
    if not self.w.forcetorque:
      for name in action:
        if name == 'dummy':
          continue
        key = name.split(':force')[0] if not self.F.use_speed else name.split(':speed')[0]
        torque = self.MOTORS_TORQUE['default']
        speed = self.SPEEDS['default']
        for mkey in self.MOTORS_TORQUE:
          if mkey in key: torque = self.MOTORS_TORQUE[mkey]
        for skey in self.SPEEDS:
          if skey in key: speed = self.SPEEDS[skey]
        if self.F.use_speed:
          self.joints[key].motorSpeed = float(speed  * np.clip(action[name], -1, 1))
        else:
          self.joints[key].motorSpeed = float(speed * np.sign(action[name]))
          self.joints[key].maxMotorTorque = float(torque * np.clip(np.abs(action[name]), 0, 1))
    else:
      # APPLY FORCE AND TORQUE
      for agent in self.w.agents:
        key = agent.name + ':root'
        body = self.dynbodies[key]
        mag = 20*action[f'{key}:force']
        if mag < 0: mag *= 0.25  # you can only go half-speed in reverse, to incentivize forward movement
        theta = 5*action[f'{key}:theta']
        fx, fy = -mag*np.sin(body.angle), mag*np.cos(body.angle)
        body_pos = body.position 
        point = body_pos# + body.localCenter
        force = (fx, fy)
        body.ApplyForce(force=force, point=point, wake=True)
        body.ApplyTorque(theta, wake=True)
    #self.end_positions = {key: A[self.dynbodies[key].position] for key in self.dynbodies}
    # RUN SIM STEP
    self.world.Step(1.0/FPS, 6*30, 2*30)
    obs = self._get_obs()
    #bodies = [self.dynbodies[key] for key in self.dynbodies if self.cname in key]
    if not self.F.walls:
      self.scroll = self.dynbodies[f'{self.cname}:root'].position.x - self.VIEWPORT_W/SCALE/2

    if self.F.reward_mode == 'env':
      reward = np.abs(self.dynbodies[f'{self.cname}:root'].linearVelocity.x)
    elif self.F.reward_mode == 'goal':
      if self.F.only_obj_goal:
        oy = self.dynbodies['object0'].position.y > 0.4
        reward = 0.1*oy
      else:
        reward = 0.0
    else:
      if self.F.num_agents == 0:
        reward = 0
      else:
        angle = np.arctan2(*obs[f'{self.cname}:root:sin', f'{self.cname}:root:cos'])
        back_penalty = (angle**2)/9
        #print(, angle)
        #import ipdb; ipdb.set_trace()
        spin_penalty = (self.dynbodies[f'{self.cname}:root'].angularVelocity**2)
        #if spin_penalty < 5.0:
        #  spin_penalty = 0.0
        spin_penalty /= 10.0
        #back_penalty = (np.abs(angle) > 2.0).astype(np.float)
        reward = -self.F.spin_penalty*spin_penalty + -self.F.back_penalty*back_penalty
        #print(reward, back_penalty, spin_penalty)
    #reward = self.dynbodies['walker0:root'].linearVelocity.x
    #reward =  self.dynbodies['walker0:root'].position.x / self.WIDTH
    #masses = [b.mass for b in bodies]
    #linvel = [b.linearVelocity.x for b in bodies]
    #momentum = (A[masses] * A[linvel]).mean()
    #reward = momentum
    #reward = np.abs(momentum)
    #info = {
    #  'crab/linvel': np.abs(self.dynbodies['crab0:root'].linearVelocity),
    #  'object/linvel': np.abs(self.dynbodies['object0'].linearVelocity),
    #}
    info = {}
    done = self.ep_t >= self.F.ep_len
    return obs.arr, reward, done, info

  def visualize_obs(self, obs, *args, **kwargs):
    if 'offset' in kwargs:
      offset = kwargs.pop('offset')
    else:
      offset = (1,0)
    entis = kwargs.get('entis', [])
    entis = [(obs, 0.8, offset)] + entis
    kwargs['entis'] = entis
    return self.render(mode='rgb_array', **kwargs)

  def render(self, mode='rgb_array', texts=[], imgs=[], shapes=[], entis=[], show_main=True, main_tilt=0.0, action=None, lines=[]):
    from envs import rendering
    if self.viewer is None:
      self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H, config=self.F)
    self.viewer.set_bounds(self.scroll, self.scale*self.VIEWPORT_W/(SCALE)+self.scroll, 0, self.scale*self.VIEWPORT_H/(SCALE))
    def draw_face(position, angle, color):
      X = 0.1; X2 = 0.2
      rot = utils.make_rot(angle); pos = position;
      # eyes
      t = rendering.Transform(translation=pos+rot.dot(A[X2, +X])); self.viewer.draw_circle(X, 20, color=color).add_attr(t)
      t = rendering.Transform(translation=pos+rot.dot(A[-X2, +X])); self.viewer.draw_circle(X, 20, color=color).add_attr(t)
      # smiley
      t = rendering.Transform(translation=pos+rot.dot(A[0.0, -X2]), rotation=angle)
      self.viewer.draw_line((0, 0), (X2, X), color=color).add_attr(t)
      self.viewer.draw_line((0, 0), (-X2, X), color=color).add_attr(t)

    for shape in shapes:
      x, y, angle, r, *color = shape
      color = 1-A[color]
      color = [*color, 1.0]
      trans = A[x,y]
      if False:
      #if angle == 0.0:
        t = rendering.Transform(translation=trans)
        self.viewer.draw_circle(r, 20, color=color).add_attr(t)
        self.viewer.draw_circle(r, 20, color=0.9*A[color], filled=False, linewidth=2).add_attr(t)
      else:
        trans = Box2D.b2Transform()
        trans.position = (x,y)
        trans.angle = angle
        path = [trans*(A[v]/(30.0*4.0)) for v in HOUSE_POLY]
        self.viewer.draw_polygon(A[path], color=color)
        path.append(path[0])
        self.viewer.draw_polyline(A[path], color=0.9*A[color], linewidth=self.VIEWPORT_H/100)

    #if len(entis) != 0: fade = np.linspace(0.8, 0.1, len(entis))
    #if len(entis) != 0: fade = np.linspace(0.8, 0.25, len(entis))
    #if len(entis) != 0: fade = np.linspace(0.75, 0.25, len(entis))
    #if len(entis) != 0: fade = np.linspace(0.1, 0.8, len(entis))
    def make_color(body, extra):
      color1, color2 = A[body.color1], A[body.color2]
      if len(extra) != 0 and extra[0] is not None:
        if extra[0] > 0.0:
          color1[0] -= extra[0]
        else:
          color1[1] -= extra[0]
        color1 = np.clip(color1, 0.0, 1.0)
      return color1, color2

    def draw_it(name, body, extra, position, angle):
      color1, color2 = make_color(body, extra)
      if len(extra) == 3:
        color1, color2 = extra[2], A[extra[2]]*0.7
      trans = Box2D.b2Transform()
      trans.position = position + (self.F.env_size*A[offset])/30.0
      trans.angle = angle
      path = [trans*v for v in body.fixtures[0].shape.vertices]
      self.viewer.draw_polygon(A[path], color=(*color1[:-1], alpha))
      path.append(path[0])
      self.viewer.draw_polyline(A[path], color=(*color2[:-1], alpha), linewidth=self.VIEWPORT_H/100)
      if 'root' in name:
        draw_face(position + (self.F.env_size*A[offset])/30.0, angle, (*color2[:-1], alpha))


    reals = {}
    fakes = {}

    for i in range(len(entis)):
      j = i
      enti, alpha, offset, *extra = entis[j]
      enti = utils.DWrap(enti, self.obs_info)

      def comp_angle(name, base_pos):
        cxy = enti[f'{name}:cx:p', f'{name}:cy:p'] - position
        base_angle = np.arctan2(*self.dynbodies[name].fixtures[0].shape.vertices[-1][::-1])
        offset_angle = np.arctan2(*cxy[::-1])
        ang = offset_angle - base_angle
        return np.arctan2(np.sin(ang), np.cos(ang))

      if len(extra) < 4 or (extra[3] is None or extra[3]):
        for agent in self.w.agents:
          name = agent.name+':root'
          root_position = position = enti[f'{name}:x:p', f'{name}:y:p']
          if self.F.all_corners:
            root_angle = angle = comp_angle(name, root_position)
          else:
            root_angle = angle = np.arctan2(*enti[f'{name}:sin', f'{name}:cos'])
          draw_it(name, self.dynbodies[name], extra, position, angle)
          base_angles = {name: root_angle}
          parent_angles = {name: root_angle}
          parent_pos = {name: root_position}
          if len(extra) > 1 and extra[1] is not None:
            act_dict = self.get_act_dict(extra[1])
            act_dict = {':'.join(key.split(':')[:2]): val for key, val in act_dict.items()}

          for ii, bj_name in enumerate(agent.joints):
            name = agent.name + ':' + bj_name
            body = self.dynbodies[name]
            if not self.F.compact_obs:
              if (float(self.F.env_version) < 0.5 and float(self.F.env_version) >= 0.3):
                if agent.joints[bj_name].limits[0] == agent.joints[bj_name].limits[1]:
                  continue
              position = enti[f'{name}:x:p', f'{name}:y:p']
              if self.F.root_offset:
                position += enti[f'{agent.name}:root:x:p', f'{agent.name}:root:y:p']
              if self.F.all_corners:
                angle = comp_angle(name, position)
              else:
                angle = np.arctan2(*enti[f'{name}:sin', f'{name}:cos'])
                if self.F.angular_offset:
                  #root_angle = np.arctan2(*enti[f'{agent.name}:root:sin', f'{agent.name}:root:cos'])
                  angle = root_angle + angle
                  angle = np.arctan2(np.sin(angle), np.cos(angle))
            else:
              joint = agent.joints[bj_name]
              parent_name = agent.name+':'+joint.parent

              jnt_ang = enti[f'{name}:angle']
              base = base_angles[parent_name]+jnt_ang
              mangle = base + joint.angle
              mangle = np.arctan2(np.sin(mangle), np.cos(mangle))
              base = np.arctan2(np.sin(base), np.cos(base))

              # parent rot
              aa_delta = A[joint.anchorA]
              pangle = base_angles[parent_name]
              bangle = parent_angles[parent_name]
              rot = utils.make_rot(bangle)
              aa_delta = rot.dot(aa_delta)
              # kid rot
              ab_delta = A[joint.anchorB]
              rot = utils.make_rot(mangle)
              ab_delta = rot.dot(ab_delta)

              position = parent_pos[parent_name]+aa_delta-ab_delta
              angle = mangle
              base_angles[name] = base
              parent_angles[name] = mangle
              parent_pos[name] = position
            fakes[name] = (position, angle)
            draw_it(name, body, extra, position, angle)
            if len(extra) > 1 and extra[1] is not None and name in act_dict:
              # draw arrows 
              pos = position + (self.F.env_size*A[offset])/30.0
              #pos = body.position
              dx = utils.make_rot(angle).dot([0, act_dict[name]])
              endpoint = pos + dx
              d = 0.25*np.abs(act_dict[name])
              a1 = pos + utils.make_rot(angle).dot([0+d, act_dict[name]-np.sign(act_dict[name])*d])
              a2 = pos + utils.make_rot(angle).dot([0-d, act_dict[name]-np.sign(act_dict[name])*d])
              self.viewer.draw_line(pos, endpoint, width=3, color=(0, 0, 1.0, 1.0))
              self.viewer.draw_line(endpoint, a1, width=3, color=(0, 0, 1.0, 1.0))
              self.viewer.draw_line(endpoint, a2, width=3, color=(0, 0, 1.0, 1.0))

      for obj in self.w.objects:
        name = obj.name
        position = enti[f'{name}:x:p', f'{name}:y:p']
        if self.F.all_corners:
          angle = comp_angle(name, position)
        else:
          angle = np.arctan2(*enti[f'{name}:sin', f'{name}:cos'])
        draw_it(name, self.dynbodies[name], extra, position, angle)

    if show_main:
      for name in self.dynbodies.keys():
        body = self.dynbodies[name]
        color1, color2 = make_color(body, [main_tilt])
        for f in body.fixtures:
          trans = f.body.transform
          if type(f.shape) is circleShape:
            t = rendering.Transform(translation=trans*f.shape.pos)
            self.viewer.draw_circle(f.shape.radius, 20, color=color1).add_attr(t)
            self.viewer.draw_circle(f.shape.radius, 20, color=color2, filled=False, linewidth=2).add_attr(t)
          else:
            reals[name] = (body.position, body.angle)
            path = [trans*v for v in f.shape.vertices]
            self.viewer.draw_polygon(A[path], color=color1)
            #self.viewer.draw_polygon(A[path]/self.scale, color=color1)
            path.append(path[0])
            self.viewer.draw_polyline(A[path], color=color2, linewidth=self.VIEWPORT_H/100)
            #self.viewer.draw_polyline(A[path]/self.scale, color=color2, linewidth=self.VIEWPORT_H/100)
          if 'root' in name:
            draw_face(body.position, body.angle, color2)
          if action is not None:
            act_dict = self.get_act_dict(action)
            if name in act_dict:
              act_dict = {':'.join(key.split(':')[:2]): val for key, val in act_dict.items()}
              # draw arrows 
              pos = self.joints[name].anchorB
              #pos = body.position
              dx = utils.make_rot(body.angle).dot([0, act_dict[name]])
              endpoint = pos + dx
              d = 0.25*np.abs(act_dict[name])
              a1 = pos + utils.make_rot(body.angle).dot([0+d, act_dict[name]-np.sign(act_dict[name])*d])
              a2 = pos + utils.make_rot(body.angle).dot([0-d, act_dict[name]-np.sign(act_dict[name])*d])
              self.viewer.draw_line(pos, endpoint, width=3, color=(0, 0, 1.0, 1.0))
              self.viewer.draw_line(endpoint, a1, width=3, color=(0, 0, 1.0, 1.0))
              self.viewer.draw_line(endpoint, a2, width=3, color=(0, 0, 1.0, 1.0))

    S = self.F.env_size/SCALE
    SS = S*self.scale
    self.viewer.draw_line((0, SS), (SS, SS), color=(0, 0, 0, 1))
    self.viewer.draw_line((SS, 0), (SS, SS), color=(0, 0, 0, 1))
    self.viewer.draw_line((0, 0), (0, SS), color=(0, 0, 0, 1))
    if self.F.special_viewer:
      # hoz
      self.viewer.draw_line((0, 1*S), (5*S,1*S), color=(0, 0, 0, 0.5))
      self.viewer.draw_line((0, 2*S), (5*S,2*S), color=(0, 0, 0, 0.5))
      # vert
      self.viewer.draw_line((1*S, 0), (1*S,3*S), color=(0, 0, 0, 0.5))
      self.viewer.draw_line((2*S, 0), (2*S,3*S), color=(0, 0, 0, 0.5))
      self.viewer.draw_line((3*S, 0), (3*S,3*S), color=(0, 0, 0, 0.5))
      self.viewer.draw_line((4*S, 0), (4*S,3*S), color=(0, 0, 0, 0.5))
      self.viewer.draw_line((5*S, 0), (5*S,3*S), color=(0, 0, 0, 0.5))

    #print()
    #for key in fakes:
    #  print(key, reals[key], fakes[key])
    #print()
    for line in lines:
      shape, color, offset = line
      xy, dxy = shape
      xy = self.F.env_size*(xy + A[offset])/30.0
      dxy = self.F.env_size*(dxy + A[offset])/30.0
      #dxy += (self.F.env_size*A[offset])/30.0
      self.viewer.draw_line(xy, dxy, color=color)

    return self.viewer.render(return_rgb_array = mode=='rgb_array', texts=texts, size=4//self.scale, imgs=imgs)[-self.F.env_size:,:int(self.F.env_wh_ratio*self.F.env_size)]

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None