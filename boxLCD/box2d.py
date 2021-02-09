from gym.envs.classic_control import rendering
from boxLCD.world_defs import FPS, SCALE, MAKERS
from boxLCD.index_env import IndexEnv
import pyglet
import time
import itertools
from collections import defaultdict
from typing import NamedTuple, List, Set, Tuple, Dict
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A

import copy
import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import utils
A = utils.A # np.array[]

class B2D(IndexEnv):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': FPS
  }

  @property
  def WIDTH(self):
    return int(self.C.env_wh_ratio * 10)

  @property
  def HEIGHT(self):
    return 10

  def __init__(self, w, C):
    """world, Hps"""
    EzPickle.__init__(self)
    self.w = w
    self.C = C
    self.SCALE = SCALE
    self.FPS = FPS
    self.scroll = 0.0
    self.VIEWPORT_H = self.C.env_size
    self.VIEWPORT_W = int(self.C.env_wh_ratio * self.C.env_size)
    self.scale = 320 / self.VIEWPORT_H
    self.SPEEDS = defaultdict(lambda: 8) if self.C.use_speed else defaultdict(lambda: 6)
    self.MOTORS_TORQUE = defaultdict(lambda: 150) if float(self.C.env_version) < 0.3 or float(self.C.env_version) >= 0.6 else defaultdict(lambda: 100)
    if float(self.C.env_version) >= 0.6:
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
      if self.C.all_corners:
        self.obs_info[f'{obj.name}:kx:p'] = A[0, self.WIDTH]
        self.obs_info[f'{obj.name}:ky:p'] = A[0, self.HEIGHT]
      else:
        self.obs_info[f'{obj.name}:cos'] = A[-1, 1]
        self.obs_info[f'{obj.name}:sin'] = A[-1, 1]

    for i in range(len(self.w.agents)):
      self.w.agents[i] = agent = MAKERS[self.C.cname](self.w.agents[i].name, SCALE, self.C)
      self.cname = self.C.cname + '0'
      self.obs_info[f'{agent.name}:root:x:p'] = A[0, self.WIDTH]
      self.obs_info[f'{agent.name}:root:y:p'] = A[0, self.HEIGHT]
      if self.C.all_corners:
        self.obs_info[f'{agent.name}:kx:p'] = A[0, self.WIDTH]
        self.obs_info[f'{agent.name}:ky:p'] = A[0, self.HEIGHT]
      else:
        self.obs_info[f'{agent.name}:root:cos'] = A[-1, 1]
        self.obs_info[f'{agent.name}:root:sin'] = A[-1, 1]

      for joint_name, joint in agent.joints.items():
        if self.C.root_offset:
          self.obs_info[f'{agent.name}:{joint_name}:x:p'] = A[-2.0, 2.0]
          self.obs_info[f'{agent.name}:{joint_name}:y:p'] = A[-2.0, 2.0]
        else:
          self.obs_info[f'{agent.name}:{joint_name}:x:p'] = A[0, self.WIDTH]
          self.obs_info[f'{agent.name}:{joint_name}:y:p'] = A[0, self.HEIGHT]

        if self.C.all_corners:
          self.obs_info[f'{agent.name}:{joint_name}:kx:p'] = A[0, self.WIDTH]
          self.obs_info[f'{agent.name}:{joint_name}:ky:p'] = A[0, self.HEIGHT]
        else:
          self.obs_info[f'{agent.name}:{joint_name}:cos'] = A[-1, 1]
          self.obs_info[f'{agent.name}:{joint_name}:sin'] = A[-1, 1]

        if joint.limits[0] != joint.limits[1]:
          # act
          if self.C.use_speed:
            self.act_info[f'{agent.name}:{joint_name}:speed'] = A[-1, 1]
          else:
            self.act_info[f'{agent.name}:{joint_name}:force'] = A[-1, 1]

    if len(self.w.agents) == 0:  # because having a zero shaped array makes things break
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
      if ur is None:
        ur = -lr
      return utils.mapto(self.np_random.uniform(lr, ur), self.obs_info[f'{namex}'])

    for agent in self.w.agents:
      color = (0.9, 0.4, 0.4, 1.0), (0.5, 0.3, 0.5, 1.0)
      # TODO: maybe create root first, and then add each of the body-joint pairs onto that.
      # this way we know where they should go. but first, let's get food.
      # like root body, then everything else is a body and a joint. joint specifies how the body attachs.
      root_body = agent.root_body

      if float(self.C.env_version) >= 0.4:
        fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits, friction=1.0)
      else:
        fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits)

      name = agent.name + ':root'
      if self.cname == 'crab0' or self.cname == 'urchin0':
        if self.C.env_wh_ratio == 2:
          root_xy = A[sample(name + ':x:p', -0.9, 0.95), sample(name + ':y:p', -0.7, -0.7)]
        elif self.C.env_wh_ratio < 1:
          rat = self.C.env_wh_ratio
          root_xy = A[sample(name + ':x:p', -0.7 * rat, 0.8 * rat), sample(name + ':y:p', -0.7, -0.70)]
        else:
          root_xy = A[sample(name + ':x:p', -0.7, 0.8), sample(name + ':y:p', -0.7, -0.70)]
      elif self.cname == 'luxo0':
        root_xy = A[sample(name + ':x:p', -0.7, 0.7), sample(name + ':y:p', -0.6, -0.60)]
      else:
        root_xy = A[sample(name + ':x:p', *agent.rangex), sample(name + ':y:p', *agent.rangey)]

      #root_xy = A[sample(name+':x:p', -0.85, -0.8), sample(name+':y:p', -0.75, -0.70)]
      #root_xy = A[sample(name+':x:p', -0.85, -0.8), sample(name+':y:p', -0.50, -0.50)]
      #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.85, 0.80)
      #root_xy = A[sample(name+':x:p', -0.7), sample(name+':y:p', -0.75, -0.70)]
      #root_xy = sample(name+':x:p', -0.85), sample(name+':y:p', -0.75, -0.70)
      def comp_angle(name, body, base_pos):
        cxy = (sample(f'{name}:kx:p'), sample(f'{name}:ky:p')) - A[base_pos]
        base_angle = np.arctan2(*body.shape.vertices[-1][::-1])
        offset_angle = np.arctan2(*cxy[::-1])
        ang = offset_angle - base_angle
        return np.arctan2(np.sin(ang), np.cos(ang))

      if self.C.all_corners:
        body = root_body
        root_angle = comp_angle(name, body, root_xy)
      else:
        root_angle = np.arctan2(sample(name + ':sin'), sample(name + ':cos'))

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
        parent_name = agent.name + ':' + joint.parent
        mangle = root_angle + joint.angle
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
            position=self.dynbodies[parent_name].position + aa_delta - ab_delta,
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
      obj_size = obj.size * self.HEIGHT / 30.00
      color = (0.5, 0.4, 0.9, 1.0), (0.3, 0.3, 0.5, 1.0)
      #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
      obj_shapes = {'circle': circleShape(radius=obj_size, pos=(0, 0)), 'box': (polygonShape(box=(obj_size, obj_size)))}
      shape_name = list(obj_shapes.keys())[np.random.randint(len(obj_shapes))] if obj.shape == 'random' else obj.shape
      shape = obj_shapes[shape_name]
      fixture = fixtureDef(shape=shape, density=obj.density, friction=obj.friction, categoryBits=obj.categoryBits, restitution=0 if shape_name == 'box' else 0.7)
      if len(self.w.agents) == 0:
        pos = A[(sample(obj.name + ':x:p', -0.90, 0.90), sample(obj.name + ':y:p', -0.90, 0.90))]
      else:
        pos = A[(sample(obj.name + ':x:p', -0.95), sample(obj.name + ':y:p', -0.9, -0.25))]

      if self.C.all_corners:
        samp = sample(obj.name + ':kx:p'), sample(obj.name + ':ky:p')
        angle = np.arctan2(*(pos - samp))
      else:
        angle = np.arctan2(sample(obj.name + ':sin'), sample(obj.name + ':cos'))

      body = self.world.CreateDynamicBody(
          position=pos,
          angle=angle,
          fixtures=fixture,
          angularDamping=0.1,
          # linearDamping=0.5,
          linearDamping=obj.damping,
      )
      body.color1, body.color2 = color
      self.dynbodies[obj.name] = body

  def reset(self, inject_obs=None):
    self.ep_t = 0
    self._destroy()
    self.statics = {}
    if self.C.walls:
      if float(self.C.env_version) <= 0.1:
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.WIDTH, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.HEIGHT)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.WIDTH, 0), (self.WIDTH, self.HEIGHT)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.HEIGHT), (self.WIDTH, self.HEIGHT)]))
      else:
        X = 0.6
        self.statics['wall1'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.WIDTH + X, 0)]))
        self.statics['wall2'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.HEIGHT + X)]))
        self.statics['wall3'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(self.WIDTH + X, 0), (self.WIDTH + X, self.HEIGHT)]))
        self.statics['wall4'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.HEIGHT + X), (self.WIDTH + X, self.HEIGHT + X)]))
    else:
      self.statics['floor'] = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(-1000 * self.WIDTH, 0), (1000 * self.WIDTH, 0)]))
    self._reset_bodies()
    #self.world.Step(0.001/FPS, 6*30, 2*30)
    if inject_obs is not None:
      def comp_angle(name, body, base_pos):
        cxy = (inject_obs[f'{name}:kx:p'], inject_obs[f'{name}:ky:p']) - A[base_pos]
        base_angle = np.arctan2(*body.shape.vertices[-1][::-1])
        offset_angle = np.arctan2(*cxy[::-1])
        ang = offset_angle - base_angle
        return np.arctan2(np.sin(ang), np.cos(ang))

      inject_obs = utils.WrappedArray(np.array(inject_obs).astype(np.float), self.obs_info)

      if len(self.w.agents) != 0:
        name = self.w.agents[0].name + ':root'
        root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

      for obj in self.w.objects:
        name = obj.name
        body = self.dynbodies[name]
        self.dynbodies[name].position = xy = inject_obs[f'{name}:x:p', f'{name}:y:p']
        if self.C.all_corners:
          import ipdb; ipdb.set_trace()  # TODO: make comp angle work with object as well
          self.dynbodies[name].angle = comp_angle(name, body, xy)
        else:
          self.dynbodies[name].angle = np.arctan2(inject_obs(name + ':sin'), inject_obs(name + ':cos'))

      for agent in self.w.agents:
        name = agent.name + ':root'
        self.dynbodies[f'{name}'].position = root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

        if self.C.all_corners:
          self.dynbodies[f'{name}'].angle = root_angle = comp_angle(name, agent.root_body, root_xy)
        else:
          self.dynbodies[f'{name}'].angle = root_angle = np.arctan2(inject_obs(name + ':sin'), inject_obs(name + ':cos'))
        parent_angles = {}
        parent_angles[name] = root_angle

        for bj_name in agent.joints:
          name = agent.name + ':' + bj_name
          body = agent.bodies[bj_name]
          joint = agent.joints[bj_name]
          if (float(self.C.env_version) < 0.5 and float(self.C.env_version) >= 0.3) and joint.limits[0] == joint.limits[1]:
            continue
          parent_name = agent.name + ':' + joint.parent
          mangle = root_angle + joint.angle
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
          if self.C.root_offset:
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[root_xy] + A[(inject_obs[name + ':x:p'], inject_obs[name + ':y:p'])]
          else:
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[(inject_obs[name + ':x:p'], inject_obs[name + ':y:p'])]

          if self.C.all_corners:
            offset_angle = comp_angle(name, agent.bodies[name.split(':')[1]], pos)
          else:
            offset_angle = np.arctan2(inject_obs[name + ':sin'], inject_obs[name + ':cos'])
            if self.C.angular_offset:
              offset_angle = root_angle + offset_angle
              offset_angle = np.arctan2(np.sin(offset_angle), np.cos(offset_angle))
          self.dynbodies[name].angle = offset_angle
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.cname}:root'].position.x - self.VIEWPORT_W / SCALE / 2
    return self._get_obs().arr

  def _get_obs(self):
    obs = utils.WrappedArray(np.zeros(self.obs_size), self.obs_info)
    for obj in self.w.objects:
      body = self.dynbodies[obj.name]
      obs[f'{obj.name}:x:p'], obs[f'{obj.name}:y:p'] = body.position
      if self.C.all_corners:
        obs[f'{obj.name}:kx:p', f'{obj.name}:ky:p'] = A[body.transform * body.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{obj.name}:cos'] = np.cos(body.angle)
        obs[f'{obj.name}:sin'] = np.sin(body.angle)

    for agent in self.w.agents:
      root = self.dynbodies[agent.name + ':root']
      obs[f'{agent.name}:root:x:p'], obs[f'{agent.name}:root:y:p'] = root_xy = root.position

      if self.C.obj_offset:
        obs[f'{obj.name}:xd:p'], obs[f'{obj.name}:yd:p'] = obs[f'{obj.name}:x:p', f'{obj.name}:y:p'] - A[root_xy]

      if self.C.all_corners:
        obs[f'{agent.name}:root:kx:p', f'{agent.name}:root:ky:p'] = A[root.transform * root.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{agent.name}:root:cos'] = np.cos(root.angle)
        obs[f'{agent.name}:root:sin'] = np.sin(root.angle)
      for joint_name, joint in agent.joints.items():
        jnt = self.joints[f'{agent.name}:{joint_name}']
        if (float(self.C.env_version) < 0.5 and float(self.C.env_version) >= 0.3) and joint.limits[0] == joint.limits[1]:
          continue
        if self.C.compact_obs:
          obs[f'{agent.name}:{joint_name}:angle'] = jnt.angle
        else:
          if self.C.root_offset:
            obs[f'{agent.name}:{joint_name}:x:p'], obs[f'{agent.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position - root_xy
          else:
            obs[f'{agent.name}:{joint_name}:x:p'], obs[f'{agent.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position
          if self.C.angular_offset:
            angle = jnt.bodyB.transform.angle - root.angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
          else:
            angle = jnt.bodyB.transform.angle
          if self.C.all_corners:
            obs[f'{agent.name}:{joint_name}:kx:p', f'{agent.name}:{joint_name}:ky:p'] = A[jnt.bodyB.transform * jnt.bodyB.fixtures[0].shape.vertices[-1]]
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
        key = name.split(':force')[0] if not self.C.use_speed else name.split(':speed')[0]
        torque = self.MOTORS_TORQUE['default']
        speed = self.SPEEDS['default']
        for mkey in self.MOTORS_TORQUE:
          if mkey in key:
            torque = self.MOTORS_TORQUE[mkey]
        for skey in self.SPEEDS:
          if skey in key:
            speed = self.SPEEDS[skey]
        if self.C.use_speed:
          self.joints[key].motorSpeed = float(speed * np.clip(action[name], -1, 1))
        else:
          self.joints[key].motorSpeed = float(speed * np.sign(action[name]))
          self.joints[key].maxMotorTorque = float(torque * np.clip(np.abs(action[name]), 0, 1))
    else:
      # APPLY FORCE AND TORQUE
      for agent in self.w.agents:
        key = agent.name + ':root'
        body = self.dynbodies[key]
        mag = 20 * action[f'{key}:force']
        if mag < 0:
          mag *= 0.25  # you can only go half-speed in reverse, to incentivize forward movement
        theta = 5 * action[f'{key}:theta']
        fx, fy = -mag * np.sin(body.angle), mag * np.cos(body.angle)
        body_pos = body.position
        point = body_pos  # + body.localCenter
        force = (fx, fy)
        body.ApplyForce(force=force, point=point, wake=True)
        body.ApplyTorque(theta, wake=True)
    #self.end_positions = {key: A[self.dynbodies[key].position] for key in self.dynbodies}
    # RUN SIM STEP
    self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
    obs = self._get_obs()
    #bodies = [self.dynbodies[key] for key in self.dynbodies if self.cname in key]
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.cname}:root'].position.x - self.VIEWPORT_W / SCALE / 2

    info = {}
    reward = 0.0 # no reward swag
    done = self.ep_t >= self.C.ep_len
    return obs.arr, reward, done, info

  def visualize_obs(self, obs, *args, **kwargs):
    if 'offset' in kwargs:
      offset = kwargs.pop('offset')
    else:
      offset = (1, 0)
    entis = kwargs.get('entis', [])
    entis = [(obs, 0.8, offset)] + entis
    kwargs['entis'] = entis
    return self.render(mode='rgb_array', **kwargs)

  def lcd_render(self):
    image = Image.new("1", (self.C.lcd_w, self.C.lcd_h))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, self.C.lcd_w, self.C.lcd_h], fill=1)
    for name, body in self.dynbodies.items():
      pos = A[body.position]
      shape = body.fixtures[0].shape
      if isinstance(shape, circleShape):
        rad = shape.radius
        topleft = (pos - rad) / A[self.WIDTH, self.HEIGHT]
        botright = (pos + rad) / A[self.WIDTH, self.HEIGHT]
        topleft = (topleft * self.C.lcd_w).astype(np.int)
        botright = (botright * self.C.lcd_w).astype(np.int)
        draw.ellipse(topleft.tolist() + botright.tolist(), fill=0)
      else:
        trans = body.transform
        points = A[[trans * v for v in shape.vertices]] / A[self.WIDTH, self.HEIGHT]
        points = (self.C.lcd_w * points).astype(np.int).tolist()
        points = tuple([tuple(xy) for xy in points])
        #points = ((10, 10), (10, 16), (16, 16), (16, 10))
        draw.polygon(points, fill=0)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    return np.array(image)

  def render(self, mode='rgb_array'):
    if self.viewer is None:
      self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
    self.viewer.set_bounds(self.scroll, self.scale * self.VIEWPORT_W / (SCALE) + self.scroll, 0, self.scale * self.VIEWPORT_H / (SCALE))

    def draw_face(position, angle, color):
      """draw a smiley face on the root body :)"""
      X = 0.1; X2 = 0.2
      rot = utils.make_rot(angle); pos = position
      # eyes
      t = rendering.Transform(translation=pos + rot.dot(A[X2, +X])); self.viewer.draw_circle(X, 20, color=color).add_attr(t)
      t = rendering.Transform(translation=pos + rot.dot(A[-X2, +X])); self.viewer.draw_circle(X, 20, color=color).add_attr(t)
      # smiley
      t = rendering.Transform(translation=pos + rot.dot(A[0.0, -X2]), rotation=angle)
      self.viewer.draw_line((0, 0), (X2, X), color=color).add_attr(t)
      self.viewer.draw_line((0, 0), (-X2, X), color=color).add_attr(t)

    for name in self.dynbodies.keys():
      body = self.dynbodies[name]
      color1, color2 = body.color1, body.color2
      for f in body.fixtures:
        trans = f.body.transform
        if type(f.shape) is circleShape:
          t = rendering.Transform(translation=trans * f.shape.pos)
          self.viewer.draw_circle(f.shape.radius, 20, color=color1).add_attr(t)
          self.viewer.draw_circle(f.shape.radius, 20, color=color2, filled=False, linewidth=2).add_attr(t)
        else:
          path = [trans * v for v in f.shape.vertices]
          self.viewer.draw_polygon(A[path], color=color1)
          path.append(path[0])
          self.viewer.draw_polyline(A[path], color=color2, linewidth=self.VIEWPORT_H / 100)
        if 'root' in name:
          draw_face(body.position, body.angle, color2)

    S = self.C.env_size / SCALE
    SS = S * self.scale
    self.viewer.draw_line((0, SS), (SS, SS), color=(0, 0, 0, 1))
    self.viewer.draw_line((SS, 0), (SS, SS), color=(0, 0, 0, 1))
    self.viewer.draw_line((0, 0), (0, SS), color=(0, 0, 0, 1))
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None