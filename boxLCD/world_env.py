from functools import partial
from gym.envs.classic_control import rendering
from boxLCD.world_defs import FPS, SCALE, ROBOT_FILLER
import time
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from boxLCD import utils
from boxLCD.viewer import Viewer
A = utils.A  # np.array[]
# ENVIRONMENT DEFAULT CONFIG
C = utils.AttrDict()

C.base_dim = 5
C.wh_ratio = 1.0
C.lcd_base = 16
C.lcd_render = 1
C.dark_mode = 0
C.use_arms = 1
C.use_images = 0
C.ep_len = 200
C.angular_offset = 0
C.root_offset = 0
C.obj_offset = 0
C.compact_obs = 0
C.use_speed = 1
C.all_contact = 1
C.all_corners = 0
C.walls = 1

class WorldEnv(gym.Env, EzPickle):
  """
  enables flexible specification of a box2d environment
  You pass in a world def and a configuration.
  Then this behaves like a regular env given that configuration.
  But it additionally has lcd_render.
  """
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': FPS
  }
  def __init__(self, world_def, _C):
    """
    args:
      world_def: description of what components you want in the world.
      C: things you might set as cmd line arguments
    """
    EzPickle.__init__(self)
    # env definitions
    self.world_def = world_def
    self.C = _C  # CONFIG
    # box2D stuff
    self.scroll = 0.0
    self.viewer = None
    self.statics = {}
    self.dynbodies = {}
    self.b2_world = Box2D.b2World(gravity=self.world_def.gravity)

    # OBSERVATION + ACTION specification
    self.obs_info = {}
    self.act_info = {}
    for obj in self.world_def.objects:
      self.obs_info[f'{obj.name}:x:p'] = A[0, self.WIDTH]
      self.obs_info[f'{obj.name}:y:p'] = A[0, self.HEIGHT]
      if self.C.all_corners:
        self.obs_info[f'{obj.name}:kx:p'] = A[0, self.WIDTH]
        self.obs_info[f'{obj.name}:ky:p'] = A[0, self.HEIGHT]
      else:
        self.obs_info[f'{obj.name}:cos'] = A[-1, 1]
        self.obs_info[f'{obj.name}:sin'] = A[-1, 1]

    for i in range(len(self.world_def.robots)):
      robot = self.world_def.robots[i]
      self.world_def.robots[i] = robot = ROBOT_FILLER[robot.type](robot, self.C)
      self.obs_info[f'{robot.name}:root:x:p'] = A[0, self.WIDTH]
      self.obs_info[f'{robot.name}:root:y:p'] = A[0, self.HEIGHT]
      if self.C.all_corners:
        self.obs_info[f'{robot.name}:kx:p'] = A[0, self.WIDTH]
        self.obs_info[f'{robot.name}:ky:p'] = A[0, self.HEIGHT]
      else:
        self.obs_info[f'{robot.name}:root:cos'] = A[-1, 1]
        self.obs_info[f'{robot.name}:root:sin'] = A[-1, 1]

      for joint_name, joint in robot.joints.items():
        if self.C.root_offset:
          self.obs_info[f'{robot.name}:{joint_name}:x:p'] = A[-2.0, 2.0]
          self.obs_info[f'{robot.name}:{joint_name}:y:p'] = A[-2.0, 2.0]
        else:
          self.obs_info[f'{robot.name}:{joint_name}:x:p'] = A[0, self.WIDTH]
          self.obs_info[f'{robot.name}:{joint_name}:y:p'] = A[0, self.HEIGHT]

        if self.C.all_corners:
          self.obs_info[f'{robot.name}:{joint_name}:kx:p'] = A[0, self.WIDTH]
          self.obs_info[f'{robot.name}:{joint_name}:ky:p'] = A[0, self.HEIGHT]
        else:
          self.obs_info[f'{robot.name}:{joint_name}:cos'] = A[-1, 1]
          self.obs_info[f'{robot.name}:{joint_name}:sin'] = A[-1, 1]

        if joint.limits[0] != joint.limits[1]:
          # act
          if self.C.use_speed:
            self.act_info[f'{robot.name}:{joint_name}:speed'] = A[-1, 1]
          else:
            self.act_info[f'{robot.name}:{joint_name}:force'] = A[-1, 1]

    if len(self.world_def.robots) == 0:  # because having a zero shaped array makes things break
      self.act_info['dummy'] = A[-1, 1]

    # take self.obs_info and self.act_info and pack them into the gym interface
    self.obs_info = utils.sortdict(self.obs_info)
    self.obs_size = len(self.obs_info)
    self.obs_keys = list(self.obs_info.keys())
    self.observation_space = spaces.Box(-1, +1, (self.obs_size,), dtype=np.float32)
    self.act_info = utils.sortdict(self.act_info)
    self.act_size = len(self.act_info)
    self.act_keys = list(self.act_info.keys())
    self.action_space = spaces.Box(-1, +1, (self.act_size,), dtype=np.float32)
    self.seed()

  @property
  def WIDTH(self):
    return int(self.C.wh_ratio * self.C.base_dim)

  @property
  def HEIGHT(self):
    return self.C.base_dim

  @property
  def VIEWPORT_H(self):
    return 30*self.HEIGHT

  @property
  def VIEWPORT_W(self):
    return 30*self.WIDTH

  @property
  def FPS(self):
    return FPS

  @property
  def SCALE(self):
    return SCALE

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _sample(self, namex, lr=-1.0, ur=None):
    if ur is None: ur = -lr
    return utils.mapto(self.np_random.uniform(lr, ur), self.obs_info[f'{namex}'])
    
  def _comp_angle(self, name, body, base_pos):
    import ipdb; ipdb.set_trace()
    cxy = (self._sample(f'{name}:kx:p'), self._sample(f'{name}:ky:p')) - A[base_pos]
    base_angle = np.arctan2(*body.shape.vertices[-1][::-1])
    offset_angle = np.arctan2(*cxy[::-1])
    ang = offset_angle - base_angle
    return np.arctan2(np.sin(ang), np.cos(ang))

  def close(self):
    if self.viewer is not None:
      self.viewer.window.close()
      self.viewer = None

  def _destroy(self):
    for name, body in {**self.statics, **self.dynbodies}.items():
      self.b2_world.DestroyBody(body)
    self.statics = {}
    self.dynbodies = {}

  def _reset_bodies(self):
    self.dynbodies = {}
    self.joints = {}

    for robot in self.world_def.robots:
      color = (0.9, 0.4, 0.4), (0.5, 0.3, 0.5)
      root_body = robot.root_body
      fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits, friction=1.0)
      name = robot.name + ':root'
      rat = self.C.wh_ratio
      root_xy = A[self._sample(name + ':x:p', *robot.rangex), self._sample(name + ':y:p', *robot.rangey)]

      if self.C.all_corners:
        body = root_body
        root_angle = self._comp_angle(name, body, root_xy)
      else:
        root_angle = np.arctan2(self._sample(name + ':sin'), self._sample(name + ':cos'))

      if not robot.rand_angle:
        root_angle = 0
      dyn = self.b2_world.CreateDynamicBody(
          position=root_xy,
          angle=root_angle,
          fixtures=fixture,
          angularDamping=robot.angularDamping,
          linearDamping=robot.linearDamping,
      )
      dyn.color1, dyn.color2 = color
      self.dynbodies[name] = dyn

      parent_angles = {}
      parent_angles[name] = root_angle

      for bj_name in robot.joints:
        name = robot.name + ':' + bj_name
        body = robot.bodies[bj_name]
        joint = robot.joints[bj_name]
        parent_name = robot.name + ':' + joint.parent
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
        dyn = self.b2_world.CreateDynamicBody(
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
            maxMotorTorque=joint.torque,
            motorSpeed=0,
            lowerAngle=joint.limits[0],
            upperAngle=joint.limits[1],
        )
        self.joints[name] = jnt = self.b2_world.CreateJoint(rjd)
        #self.joints[name].bodyB.transform.angle = self._sample(name+':theta')

    for obj in self.world_def.objects:
      obj_size = obj.size
      color = (0.5, 0.4, 0.9), (0.3, 0.3, 0.5)
      #fixture = fixtureDef(shape=circleShape(radius=1.0), density=1)
      obj_shapes = {'circle': circleShape(radius=obj_size, pos=(0, 0)), 'box': (polygonShape(box=(obj_size, obj_size)))}
      shape_name = list(obj_shapes.keys())[np.random.randint(len(obj_shapes))] if obj.shape == 'random' else obj.shape
      shape = obj_shapes[shape_name]
      fixture = fixtureDef(shape=shape, density=obj.density, friction=obj.friction, categoryBits=obj.categoryBits, restitution=0 if shape_name == 'box' else 0.7)
      if len(self.world_def.robots) == 0:
        pos = A[(self._sample(obj.name + ':x:p', -0.90, 0.90), self._sample(obj.name + ':y:p', -0.90, 0.90))]
      else:
        pos = A[(self._sample(obj.name + ':x:p', -0.95), self._sample(obj.name + ':y:p', -0.9, -0.25))]

      if self.C.all_corners:
        samp = self._sample(obj.name + ':kx:p'), self._sample(obj.name + ':ky:p')
        angle = np.arctan2(*(pos - samp))
      else:
        angle = np.arctan2(self._sample(obj.name + ':sin'), self._sample(obj.name + ':cos'))

      body = self.b2_world.CreateDynamicBody(
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
      X = 0.0
      self.statics['wall1'] = self.b2_world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.WIDTH + X, 0)]))
      self.statics['wall2'] = self.b2_world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (0, self.HEIGHT + X)]))
      self.statics['wall3'] = self.b2_world.CreateStaticBody(shapes=edgeShape(vertices=[(self.WIDTH + X, 0), (self.WIDTH + X, self.HEIGHT)]))
      self.statics['wall4'] = self.b2_world.CreateStaticBody(shapes=edgeShape(vertices=[(0, self.HEIGHT + X), (self.WIDTH + X, self.HEIGHT + X)]))
    else:
      self.statics['floor'] = self.b2_world.CreateStaticBody(shapes=edgeShape(vertices=[(-1000 * self.WIDTH, 0), (1000 * self.WIDTH, 0)]))
    self._reset_bodies()
    #self.b2_world.Step(0.001/FPS, 6*30, 2*30)
    if inject_obs is not None:
      inject_obs = utils.NamedArray(np.array(inject_obs).astype(np.float), self.obs_info)

      if len(self.world_def.robots) != 0:
        name = self.world_def.robots[0].name + ':root'
        root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

      for obj in self.world_def.objects:
        name = obj.name
        body = self.dynbodies[name]
        self.dynbodies[name].position = xy = inject_obs[f'{name}:x:p', f'{name}:y:p']
        if self.C.all_corners:
          import ipdb; ipdb.set_trace()  # TODO: make comp angle work with object as well
          self.dynbodies[name].angle = self._comp_angle(name, body, xy)
        else:
          self.dynbodies[name].angle = np.arctan2(inject_obs(name + ':sin'), inject_obs(name + ':cos'))

      for robot in self.world_def.robots:
        name = robot.name + ':root'
        self.dynbodies[f'{name}'].position = root_xy = inject_obs[f'{name}:x:p', f'{name}:y:p']

        if self.C.all_corners:
          self.dynbodies[f'{name}'].angle = root_angle = self._comp_angle(name, robot.root_body, root_xy)
        else:
          self.dynbodies[f'{name}'].angle = root_angle = np.arctan2(inject_obs(name + ':sin'), inject_obs(name + ':cos'))
        parent_angles = {}
        parent_angles[name] = root_angle

        for bj_name in robot.joints:
          name = robot.name + ':' + bj_name
          body = robot.bodies[bj_name]
          joint = robot.joints[bj_name]
          parent_name = roborobote + ':' + joint.parent
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
            offset_angle = self._comp_angle(name, robot.bodies[name.split(':')[1]], pos)
          else:
            offset_angle = np.arctan2(inject_obs[name + ':sin'], inject_obs[name + ':cos'])
            if self.C.angular_offset:
              offset_angle = root_angle + offset_angle
              offset_angle = np.arctan2(np.sin(offset_angle), np.cos(offset_angle))
          self.dynbodies[name].angle = offset_angle
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.world_def.robots[0].type}0:root'].position.x - self.VIEWPORT_W / SCALE / 2
    return self._get_obs().arr

  def _get_obs(self):
    obs = utils.NamedArray(np.zeros(self.obs_size), self.obs_info)
    # GRAB OBJECT INFO
    for obj in self.world_def.objects:
      body = self.dynbodies[obj.name]
      obs[f'{obj.name}:x:p'], obs[f'{obj.name}:y:p'] = body.position
      if self.C.all_corners:
        obs[f'{obj.name}:kx:p', f'{obj.name}:ky:p'] = A[body.transform * body.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{obj.name}:cos'] = np.cos(body.angle)
        obs[f'{obj.name}:sin'] = np.sin(body.angle)
    # GRAB ROBOT INFO
    for robot in self.world_def.robots:
      root = self.dynbodies[robot.name + ':root']
      obs[f'{robot.name}:root:x:p'], obs[f'{robot.name}:root:y:p'] = root_xy = root.position
      if self.C.obj_offset:
        obs[f'{obj.name}:xd:p'], obs[f'{obj.name}:yd:p'] = obs[f'{obj.name}:x:p', f'{obj.name}:y:p'] - A[root_xy]
      if self.C.all_corners:
        obs[f'{robot.name}:root:kx:p', f'{robot.name}:root:ky:p'] = A[root.transform * root.fixtures[0].shape.vertices[-1]]
      else:
        obs[f'{robot.name}:root:cos'] = np.cos(root.angle)
        obs[f'{robot.name}:root:sin'] = np.sin(root.angle)
      for joint_name, joint in robot.joints.items():
        jnt = self.joints[f'{robot.name}:{joint_name}']
        if self.C.compact_obs:
          obs[f'{robot.name}:{joint_name}:angle'] = jnt.angle
        else:
          if self.C.root_offset:
            obs[f'{robot.name}:{joint_name}:x:p'], obs[f'{robot.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position - root_xy
          else:
            obs[f'{robot.name}:{joint_name}:x:p'], obs[f'{robot.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position
          if self.C.angular_offset:
            angle = jnt.bodyB.transform.angle - root.angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
          else:
            angle = jnt.bodyB.transform.angle
          if self.C.all_corners:
            obs[f'{robot.name}:{joint_name}:kx:p', f'{robot.name}:{joint_name}:ky:p'] = A[jnt.bodyB.transform * jnt.bodyB.fixtures[0].shape.vertices[-1]]
          else:
            obs[f'{robot.name}:{joint_name}:cos'] = np.cos(angle)
            obs[f'{robot.name}:{joint_name}:sin'] = np.sin(angle)
    return obs

  def step(self, action):
    self.ep_t += 1
    action = utils.NamedArray(action, self.act_info, do_map=True)
    # JOINT CONTROL
    for robot in self.world_def.robots:
      for jname, joint in robot.joints.items():
        name = f'{robot.name}:{jname}'
        if joint.limits[0] == joint.limits[1]: continue  # joint that doesn't move
        if self.C.use_speed:
          self.joints[name].motorSpeed = float(joint.speed * np.clip(action[name + ':speed'], -1, 1))
        else:
          self.joints[name].motorSpeed = float(joint.speed * np.sign(action[name + ':torque']))
          self.joints[name].maxMotorTorque = float(joint.torque * np.clip(np.abs(action[name + ':torque']), 0, 1))
    # RUN SIM STEP
    self.b2_world.Step(1.0 / FPS, 6 * 30, 2 * 30)
    obs = self._get_obs()
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.world_def.robots[0].type}0:root'].position.x - self.VIEWPORT_W / SCALE / 2
    info = {}
    reward = 0.0  # no reward swag
    done = self.ep_t >= self.C.ep_len
    return obs.arr, reward, done, info

  def lcd_render(self, width=None, height=None, pretty=False):
    """render the env using PIL at potentially very low resolution
    """
    if width is None and height is None:
      width = int(self.C.lcd_base*self.C.wh_ratio)
      height = self.C.lcd_base

    if pretty:
      width *= 8
      height *= 8
      mode = "RGB"
      backgrond = (1,1,1)
    else:
      mode = "1"
      backgrond = 1

    image = Image.new(mode, (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, height], fill=backgrond)
    for name, body in self.dynbodies.items():
      pos = A[body.position]
      shape = body.fixtures[0].shape
      if pretty:
        color = tuple([int(255.0*(1-x)) for x in body.color1])
        outline = tuple([int(255.0*(1-x)) for x in body.color2])
        linew = 1
      else:
        color = 0
        outline = None
        linew = 1

      if isinstance(shape, circleShape):
        rad = shape.radius
        topleft = (pos - rad) / self.WIDTH
        botright = (pos + rad) / self.WIDTH
        topleft = (topleft * width)
        botright = (botright * width)
        draw.ellipse(topleft.tolist() + botright.tolist(), fill=color, outline=outline, width=linew)
      else:
        trans = body.transform
        points = A[[trans * v for v in shape.vertices]] / self.WIDTH
        points = (width * points).tolist()
        points = tuple([tuple(xy) for xy in points])
        draw.polygon(points, fill=color, outline=outline)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    lcd = np.array(image)
    return lcd
    #TODO: deal with scrolling

  def render(self, mode='rgb_array'):
    width = int(self.C.lcd_base*self.C.wh_ratio)
    height = self.C.lcd_base
    lcd = self.lcd_render(width, height)
    if mode == 'rgb_array':
      return lcd
    elif mode == 'human':
      # use a pyglet viewer to show the images to the user in realtime.
      if self.viewer is None:
        self.viewer = Viewer(width*8, height*8, self.C)
      high_res = 255*self.lcd_render(width, height, pretty=True).astype(np.uint8)
      if False:
        high_res = 255*high_res[...,None].astype(np.uint8).repeat(3,-1)
      low_res = 255*lcd.astype(np.uint8)[...,None].repeat(8, 0).repeat(8, 1).repeat(3,2)
      img = np.concatenate([high_res, np.zeros_like(low_res)[:,:2], low_res], axis=1)
      self.viewer.render(img)
      return lcd

