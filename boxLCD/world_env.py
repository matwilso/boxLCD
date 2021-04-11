from functools import partial
from gym.envs.classic_control import rendering
from boxLCD.world_defs import SCALE, ROBOT_FILLER
import time
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import gym
from gym.utils import seeding, EzPickle
from boxLCD import utils
from boxLCD.viewer import Viewer
A = utils.A  # np.array[]


# THIS IS AN ABSTRACT CLASS ALL OF THE LOGIC FOR SIMULATION
# SPECIFIC INSTANCES ARE DESCRIBED IN envs.py, WHERE SPECIFIC WORLDS ARE DEFINED
class WorldEnv(gym.Env, EzPickle):
  """
  enables flexible specification of a box2d environment
  You pass in a world def and a configuration.
  Then this behaves like a regular env given that configuration.
  But it additionally has lcd_render.
  """
  metadata = {
      'render.modes': ['human', 'rgb_array'],
  }
  # ENVIRONMENT DEFAULT CONFIG
  ENV_DC = utils.AttrDict()
  ENV_DC.base_dim = 5  # base size of box2D physics world
  ENV_DC.lcd_base = 16  # base size of lcd rendered image. this represents the height. width = wh_ratio*height
  ENV_DC.wh_ratio = 2.0  # width:height ratio of the world and images
  ENV_DC.ep_len = 200  # length to run episode before done timeout
  # settings for different obs and action spaces
  ENV_DC.angular_offset = 0  # compute joint angular offsets from robot roots
  ENV_DC.root_offset = 0  # compute position offsets from root
  ENV_DC.compact_obs = 0  # use compact joint angle space instead of joint positions and sin+cos of theta
  ENV_DC.use_speed = 1  # use velocity control vs. torque control
  ENV_DC.all_corners = 0  # use corner keypoint obs instead of sin+cos of theta
  ENV_DC.walls = 1  # bound the environment with walls on both sides
  ENV_DC.debug = 0
  ENV_DC.fps = 10

  def __init__(self, world_def, C={}):
    """
    args:
      world_def: description of what components you want in the world.
      C: things you might set as cmd line arguments
    """
    EzPickle.__init__(self)
    # env definitions
    self.world_def = world_def
    # CONFIG
    self.C = utils.AttrDict(self.ENV_DC)
    if not isinstance(C, dict):
      C = C.__dict__
    for key in C:
      self.C[key] = C[key]  # update with what gets passed in
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
    # partial observation
    self.pobs_keys = utils.nfiltlist(self.obs_keys, 'object')
    self.pobs_size = len(self.pobs_keys)
    self.pobs_idxs = [self.obs_keys.index(x) for x in self.pobs_keys]
    # observation is a dict space
    spaces = {}
    spaces['full_state'] = gym.spaces.Box(-1, +1, (self.obs_size,), dtype=np.float32)
    # pstate = partial state
    if self.pobs_size == 0:
      spaces['pstate'] = gym.spaces.Box(-1, +1, (1,), dtype=np.float32)
    else:
      spaces['pstate'] = gym.spaces.Box(-1, +1, (self.pobs_size,), dtype=np.float32)
    spaces['lcd'] = gym.spaces.Box(0, 1, (self.C.lcd_base, int(self.C.lcd_base * self.C.wh_ratio)), dtype=np.bool)
    self.observation_space = gym.spaces.Dict(spaces)

    self.act_info = utils.sortdict(self.act_info)
    self.act_size = len(self.act_info)
    self.act_keys = list(self.act_info.keys())
    self.action_space = gym.spaces.Box(-1, +1, (self.act_size,), dtype=np.float32)
    self.seed()

  @property
  def WIDTH(self):
    return int(self.C.wh_ratio * self.C.base_dim)

  @property
  def HEIGHT(self):
    return self.C.base_dim

  @property
  def VIEWPORT_H(self):
    return 30 * self.HEIGHT

  @property
  def VIEWPORT_W(self):
    return 30 * self.WIDTH

  @property
  def FPS(self):
    return self.C.fps

  @property
  def SCALE(self):
    return SCALE

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _sample(self, namex, lr=-1.0, ur=None):
    if ur is None:
      ur = -lr
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
    # ROBOT
    for robot in self.world_def.robots:
      color = (0.9, 0.4, 0.4), (0.5, 0.3, 0.5)
      root_body = robot.root_body
      fixture = fixtureDef(shape=root_body.shape, density=1.0 if root_body.density is None else root_body.density, categoryBits=root_body.categoryBits, maskBits=root_body.maskBits, friction=1.0)
      name = robot.name + ':root'
      rangex = 1 - (2 * robot.bound / self.WIDTH)
      rangey = 1 - (2 * robot.bound / self.HEIGHT)
      root_xy = A[self._sample(name + ':x:p', -rangex, rangex), self._sample(name + ':y:p', -rangey, -rangey)]

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

      # PUT ALL THE JOINT ON THE ROOT AND ENSURE CORRECT ANGLES
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

    # OBJECT
    for obj in self.world_def.objects:
      # MAKE SHAPE AND SIZE
      obj_size = obj.size
      obj_shapes = {'circle': circleShape(radius=obj_size, pos=(0, 0)), 'box': (polygonShape(box=(obj_size, obj_size)))}
      shape_name = list(obj_shapes.keys())[np.random.randint(len(obj_shapes))] if obj.shape == 'random' else obj.shape
      shape = obj_shapes[shape_name]
      if obj.restitution is None:
        restitution = 0 if shape_name == 'box' else 0.7
      else:
        restitution = obj.restitution
      fixture = fixtureDef(shape=shape, density=obj.density, friction=obj.friction, categoryBits=obj.categoryBits, restitution=restitution)
      # SAMPLE POSITION. KEEP THE OBJECT IN THE BOUNDS OF THE ARENA
      if obj.rangex is None:
        rangex = 1 - (2 * obj_size / self.WIDTH)
      if obj.rangey is None:
        rangey = 1 - (2 * obj_size / self.HEIGHT)
      if len(self.world_def.robots) == 0:
        pos = A[(self._sample(obj.name + ':x:p', -rangex, rangex), self._sample(obj.name + ':y:p', -rangey, rangey))]
      else:
        pos = A[(self._sample(obj.name + ':x:p', -rangex, rangex), self._sample(obj.name + ':y:p', -rangey, -0.25))]
      # ANGLE OR NOT
      if obj.rand_angle:
        if self.C.all_corners:
          samp = self._sample(obj.name + ':kx:p'), self._sample(obj.name + ':ky:p')
          angle = np.arctan2(*(pos - samp))
        else:
          angle = np.arctan2(self._sample(obj.name + ':sin'), self._sample(obj.name + ':cos'))
      else:
        angle = 0
      # CREATE OBJ BODY
      body = self.b2_world.CreateDynamicBody(
          position=pos,
          angle=angle,
          fixtures=fixture,
          linearDamping=obj.linearDamping,
          angularDamping=obj.angularDamping,
      )
      body.color1, body.color2 = (0.5, 0.4, 0.9), (0.3, 0.3, 0.5)
      self.dynbodies[obj.name] = body

  def reset(self, full_state=None, pstate=None):
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
    if pstate is not None:
      assert pstate.shape[-1] == self.observation_space.spaces['pstate'].shape[-1], f'invalid shape for pstate {pstate.shape} {self.observation_space.spaces["pstate"]}'
      full_state = np.zeros(self.observation_space.spaces['full_state'].shape)
      full_state[self.pobs_idxs] = pstate
    if full_state is not None:
      full_state = utils.NamedArray(np.array(full_state).astype(np.float), self.obs_info)

      if len(self.world_def.robots) != 0:
        name = self.world_def.robots[0].name + ':root'
        root_xy = full_state[f'{name}:x:p', f'{name}:y:p']

      for obj in self.world_def.objects:
        name = obj.name
        body = self.dynbodies[name]
        self.dynbodies[name].position = xy = full_state[f'{name}:x:p', f'{name}:y:p']
        if self.C.all_corners:
          import ipdb; ipdb.set_trace()  # TODO: make comp angle work with object as well
          self.dynbodies[name].angle = self._comp_angle(name, body, xy)
        else:
          self.dynbodies[name].angle = np.arctan2(full_state(name + ':sin'), full_state(name + ':cos'))

      for robot in self.world_def.robots:
        name = robot.name + ':root'
        self.dynbodies[f'{name}'].position = root_xy = full_state[f'{name}:x:p', f'{name}:y:p']

        if self.C.all_corners:
          self.dynbodies[f'{name}'].angle = root_angle = self._comp_angle(name, robot.root_body, root_xy)
        else:
          self.dynbodies[f'{name}'].angle = root_angle = np.arctan2(full_state(name + ':sin'), full_state(name + ':cos'))
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
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[root_xy] + A[(full_state[name + ':x:p'], full_state[name + ':y:p'])]
          else:
            self.dynbodies[name].position = self.joints[name].bodyB.transform.position = pos = A[(full_state[name + ':x:p'], full_state[name + ':y:p'])]

          if self.C.all_corners:
            offset_angle = self._comp_angle(name, robot.bodies[name.split(':')[1]], pos)
          else:
            offset_angle = np.arctan2(full_state[name + ':sin'], full_state[name + ':cos'])
            if self.C.angular_offset:
              offset_angle = root_angle + offset_angle
              offset_angle = np.arctan2(np.sin(offset_angle), np.cos(offset_angle))
          self.dynbodies[name].angle = offset_angle
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.world_def.robots[0].type}0:root'].position.x - self.VIEWPORT_W / SCALE / 2
    return self._get_obs()

  def _get_obs(self):
    full_state = utils.NamedArray(np.zeros(self.obs_size), self.obs_info)
    # GRAB OBJECT INFO
    for obj in self.world_def.objects:
      body = self.dynbodies[obj.name]
      full_state[f'{obj.name}:x:p'], full_state[f'{obj.name}:y:p'] = body.position
      if self.C.all_corners:
        full_state[f'{obj.name}:kx:p', f'{obj.name}:ky:p'] = A[body.transform * body.fixtures[0].shape.vertices[-1]]
      else:
        full_state[f'{obj.name}:cos'] = np.cos(body.angle)
        full_state[f'{obj.name}:sin'] = np.sin(body.angle)
    # GRAB ROBOT INFO
    for robot in self.world_def.robots:
      root = self.dynbodies[robot.name + ':root']
      full_state[f'{robot.name}:root:x:p'], full_state[f'{robot.name}:root:y:p'] = root_xy = root.position
      if self.C.all_corners:
        full_state[f'{robot.name}:root:kx:p', f'{robot.name}:root:ky:p'] = A[root.transform * root.fixtures[0].shape.vertices[-1]]
      else:
        full_state[f'{robot.name}:root:cos'] = np.cos(root.angle)
        full_state[f'{robot.name}:root:sin'] = np.sin(root.angle)
      for joint_name, joint in robot.joints.items():
        jnt = self.joints[f'{robot.name}:{joint_name}']
        if self.C.compact_obs:
          full_state[f'{robot.name}:{joint_name}:angle'] = jnt.angle
        else:
          if self.C.root_offset:
            full_state[f'{robot.name}:{joint_name}:x:p'], full_state[f'{robot.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position - root_xy
          else:
            full_state[f'{robot.name}:{joint_name}:x:p'], full_state[f'{robot.name}:{joint_name}:y:p'] = jnt.bodyB.transform.position
          if self.C.angular_offset:
            angle = jnt.bodyB.transform.angle - root.angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
          else:
            angle = jnt.bodyB.transform.angle
          if self.C.all_corners:
            full_state[f'{robot.name}:{joint_name}:kx:p', f'{robot.name}:{joint_name}:ky:p'] = A[jnt.bodyB.transform * jnt.bodyB.fixtures[0].shape.vertices[-1]]
          else:
            full_state[f'{robot.name}:{joint_name}:cos'] = np.cos(angle)
            full_state[f'{robot.name}:{joint_name}:sin'] = np.sin(angle)

    full_state = full_state.arr
    pstate = full_state[self.pobs_idxs] if self.pobs_size != 0 else np.zeros(1)
    return {'full_state': full_state, 'pstate': pstate, 'lcd': self.lcd_render()}

  def step(self, action):
    self.ep_t += 1
    action = utils.NamedArray(action, self.act_info, do_map=True)
    # JOINT CONTROL
    for robot in self.world_def.robots:
      for jname, joint in robot.joints.items():
        name = f'{robot.name}:{jname}'
        if joint.limits[0] == joint.limits[1]:
          continue  # joint that doesn't move
        if self.C.use_speed:
          self.joints[name].motorSpeed = float(joint.speed * np.clip(action[name + ':speed'], -1, 1))
        else:
          self.joints[name].motorSpeed = float(joint.speed * np.sign(action[name + ':torque']))
          self.joints[name].maxMotorTorque = float(joint.torque * np.clip(np.abs(action[name + ':torque']), 0, 1))
    # RUN SIM STEP
    self.b2_world.Step(1.0 / (self.FPS*2), 6 * 30, 2 * 30)
    self.b2_world.Step(1.0 / (self.FPS*2), 6 * 30, 2 * 30)
    if not self.C.walls:
      self.scroll = self.dynbodies[f'{self.world_def.robots[0].type}0:root'].position.x - self.VIEWPORT_W / SCALE / 2
    reward = 0.0  # no reward swag
    done = self.ep_t >= self.C.ep_len
    info = {'timeout': done}
    return self._get_obs(), reward, done, info

  def lcd_render(self, width=None, height=None, pretty=False):
    """render the env using PIL at potentially very low resolution

    # TODO: deal with scrolling
    """
    if width is None and height is None:
      width = int(self.C.lcd_base * self.C.wh_ratio)
      height = self.C.lcd_base
    if pretty:
      mode = "RGB"
      backgrond = (1, 1, 1)
    else:
      mode = "1"
      backgrond = 1

    image = Image.new(mode, (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, height], fill=backgrond)
    for body in self.dynbodies.values():
      pos = A[body.position]
      shape = body.fixtures[0].shape
      if pretty:
        color = tuple([int(255.0 * (1 - x)) for x in body.color1])
        outline = tuple([int(255.0 * (1 - x)) for x in body.color2])
      else:
        color = 0
        outline = None

      if isinstance(shape, circleShape):
        rad = shape.radius
        topleft = (pos - rad) / self.WIDTH
        botright = (pos + rad) / self.WIDTH
        topleft = (topleft * width)
        botright = (botright * width)
        draw.ellipse(topleft.tolist() + botright.tolist(), fill=color, outline=outline)
      else:
        trans = body.transform
        points = A[[trans * v for v in shape.vertices]] / self.WIDTH
        points = (width * points).tolist()
        points = tuple([tuple(xy) for xy in points])
        draw.polygon(points, fill=color, outline=outline)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    lcd = np.asarray(image)
    if lcd.dtype == np.bool:
      lcd = lcd.astype(np.float).astype(np.bool)  # fix bug where PIL produces a bool(xFF) instead of a bool(0x01)
    if pretty:
      lcd = 255 - lcd
    return lcd

  def render(self, mode='rgb_array', pretty=False, return_pyglet_view=False):
    width = int(self.C.lcd_base * self.C.wh_ratio)
    height = self.C.lcd_base
    lcd = self.lcd_render(width, height, pretty=pretty)
    if mode == 'rgb_array':
      return lcd
    elif mode == 'human':
      # use a pyglet viewer to show the images to the user in realtime.
      if self.viewer is None:
        self.viewer = Viewer(width * 8, height * 8, self.C)
      high_res = self.lcd_render(width * 8, height * 8, pretty=True).astype(np.uint8)
      if False:
        high_res = 255 * high_res[..., None].astype(np.uint8).repeat(3, -1)
      low_res = 255 * lcd.astype(np.uint8)[..., None].repeat(8, 0).repeat(8, 1).repeat(3, 2)
      img = np.concatenate([high_res, np.zeros_like(low_res)[:, :2], low_res], axis=1)
      out = self.viewer.render(img, return_rgb_array=return_pyglet_view)
      if not return_pyglet_view:
        out = lcd
      return out
