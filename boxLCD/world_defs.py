from collections import defaultdict
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)
from typing import NamedTuple, List, Set, Tuple, Dict
from boxLCD import utils
A = utils.A

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

# STRUCTS THAT ARE USED TO DEFINE A WORLD
class Object(NamedTuple):
  name: str
  shape: str = 'box'
  size: float = 0.5
  damping: float = 0.0
  density: float = 1.0
  friction: float = 0.5
  restitution: float = None
  categoryBits: int = 0x0110
  rand_angle: int = 1
  rangex: Tuple[float, float] = None
  rangey: Tuple[float, float] = None

# ROBOT PARTS
class Body(NamedTuple):
  shape: polygonShape
  density: float = 1
  maskBits: int = 0x001
  categoryBits: int = 0x0020
  friction: float = 1.0

class Joint(NamedTuple):
  parent: str
  angle: float
  anchorA: list
  anchorB: list
  limits: List[float]
  limited: bool = True
  speed: float = 8
  torque: float = 150

class Robot(NamedTuple):
  type: str
  name: str
  root_body: Body = None
  bodies: Dict[str, Body] = None
  joints: Dict[str, Joint] = None
  rand_angle: int = 0
  angularDamping: float = 0
  linearDamping: float = 0
  bound: float = 1.5  # spatial extent of robot. must be set so we don't start robot stuck in a wall

# FULL WORLD DEF
class WorldDef(NamedTuple):
  robots: List[Robot] = []
  objects: List[Object] = []
  gravity: List[float] = [0, -9.81]
  forcetorque: int = 0


# THESE TAKE A PARTIAL ROBOT DESCRIPTION AND BUILD ALL THE JOINTS AND STUFF.
ROBOT_FILLER = {}
def register(name):
  def _reg(func):
    ROBOT_FILLER[name] = func
    def _thunk(*args, **kwargs):
      return func(*args, **kwargs)
    return _thunk
  return _reg

@register('urchin')
def make_urchin(robot, C):
  LEG_W, LEG_H = 8 / SCALE, 40 / SCALE
  SHAPES = {}
  SHAPES['root'] = circleShape(radius=0.8 * LEG_W)
  SHAPES['leg'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  root_body = Body(SHAPES['root'])
  bodies = {
      'aleg': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'bleg': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'cleg': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
  }
  joints = {
      'aleg': Joint('root', 0.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=True),
      'bleg': Joint('root', 2.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=True),
      'cleg': Joint('root', 4.2, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=True),
  }
  return Robot(type=robot.type, name=robot.name, root_body=root_body, bodies=bodies, joints=joints, rand_angle=1, bound=1.5)

# urchin is the best robot. the rest need some work.
# it's actually a bit hard to design a robot that can do interesting things in 2D space.
# i guess i maybe oughta have some top down views, but then how do you control with joints?
# could have a roomba with a gripper or something like that. or a reacher arm.

@register('crab')
def make_crab(robot, C):
  SPEEDS = defaultdict(lambda: 8)
  MOTORS_TORQUE = defaultdict(lambda: 150)
  SPEEDS = defaultdict(lambda: 6)
  SPEEDS['hip'] = 10
  SPEEDS['knee'] = 10
  MOTORS_TORQUE['hip'] = 150
  MOTORS_TORQUE['knee'] = 150
  # TODO: make armless crab version.
  VERT = 12 / SCALE
  SIDE = 20 / SCALE
  LEG_W, LEG_H = 8 / SCALE, 20 / SCALE
  LL_H = 20 / SCALE
  #LEG_W, LEG_H = 8/SCALE, 16/SCALE
  ARM_W, ARM_H = 8 / SCALE, 20 / SCALE
  CLAW_W, CLAW_H = 4 / SCALE, 16 / SCALE
  CRAB_POLY = 0.9 * A[(-25, +0), (-20, +16), (+20, +16), (+25, +0), (+20, -16), (-20, -16)]
  SHAPES = {}
  SHAPES['root'] = polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in CRAB_POLY])
  #SHAPES['root'] = circleShape(radius=20/SCALE)
  SHAPES['arm'] = polygonShape(box=(ARM_W / 2, ARM_H / 2))
  SHAPES['hip'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  SHAPES['knee'] = polygonShape(box=(0.8 * LEG_W / 2, LL_H / 2))
  SHAPES['claw'] = polygonShape(box=(CLAW_W / 2, CLAW_H / 2))
  SHAPES['foot'] = polygonShape(box=(0.8 * LEG_W / 2, LL_H / 4))

  baseMask = 0x001
  clawMask = 0x011
  categoryBits: int = 0x0020
  if True:
    maskBits = 0x001
    baseMask = 0x001
  else:
    maskBits = clawMask
    baseMask = clawMask

  bodies = {
      'lhip': Body(SHAPES['hip'], maskBits=baseMask),
      'lknee': Body(SHAPES['knee'], maskBits=baseMask),
      'rhip': Body(SHAPES['hip'], maskBits=baseMask),
      'rknee': Body(SHAPES['knee'], maskBits=baseMask),
      'lshoulder': Body(SHAPES['arm'], maskBits=maskBits),
      'lelbow': Body(SHAPES['arm'], maskBits=maskBits),
      'rshoulder': Body(SHAPES['arm'], maskBits=maskBits),
      'relbow': Body(SHAPES['arm'], maskBits=maskBits),
      # left claw
      'llclaw0': Body(SHAPES['claw'], maskBits=clawMask),
      'llclaw1': Body(SHAPES['claw'], maskBits=clawMask),
      'lrclaw0': Body(SHAPES['claw'], maskBits=clawMask),
      'lrclaw1': Body(SHAPES['claw'], maskBits=clawMask),
      # right claw
      'rlclaw0': Body(SHAPES['claw'], maskBits=clawMask),
      'rlclaw1': Body(SHAPES['claw'], maskBits=clawMask),
      'rrclaw0': Body(SHAPES['claw'], maskBits=clawMask),
      'rrclaw1': Body(SHAPES['claw'], maskBits=clawMask),
  }
  joints = {
      # legs
      'lhip': Joint('root', -0.5, (-SIDE, -VERT), (0, LEG_H / 2), [-1.5, 0.5]),
      'rhip': Joint('root', 0.5, (SIDE, -VERT), (0, LEG_H / 2), [0.5, 1.5]),
      'lknee': Joint('lhip', 0.5, (0, -LEG_H / 2), (0, LL_H / 2), [-0.5, 0.5]),
      'rknee': Joint('rhip', -0.5, (0, -LEG_H / 2), (0, LL_H / 2), [-0.5, 0.5]),
  }
  joints.update(**{
      # arms
      'lshoulder': Joint('root', 2.0, (-SIDE, VERT), (0, -ARM_H / 2), [-3.0, 3.0], limited=False),
      'rshoulder': Joint('root', -2.0, (SIDE, VERT), (0, -ARM_H / 2), [-3.0, 3.0], limited=False),
      'lelbow': Joint('lshoulder', 3.0, (0, ARM_H / 2), (0, -ARM_H / 2), [-2.0, 2.0], limited=False),
      'relbow': Joint('rshoulder', -3.0, (0, ARM_H / 2), (0, -ARM_H / 2), [-2.0, 2.0], limited=False),
      # left claw
      'llclaw0': Joint('lelbow', 2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-2.0, 1.0]),
      'llclaw1': Joint('llclaw0', 3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
      'lrclaw0': Joint('lelbow', -2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-1.0, 2.0]),
      'lrclaw1': Joint('lrclaw0', -3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
      # right claw
      'rlclaw0': Joint('relbow', 2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-2.0, 1.0]),
      'rlclaw1': Joint('rlclaw0', 3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
      'rrclaw0': Joint('relbow', -2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-1.0, 2.0]),
      'rrclaw1': Joint('rrclaw0', -3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
  },)
  return Robot(type=robot.type, name=robot.name, root_body=Body(SHAPES['root'], density=1.0, maskBits=baseMask, categoryBits=categoryBits), bodies=bodies, joints=joints, bound=2.0)

# TODO: make armed walker
@register('walker')
def make_walker(robot, C):
  LEG_DOWN = -6 / SCALE
  LEG_W, LEG_H = 10 / SCALE, 24 / SCALE
  ARM_W, ARM_H = 8 / SCALE, 20 / SCALE
  CLAW_W, CLAW_H = 6 / SCALE, 16 / SCALE
  HULL_POLY = 0.8 * A[(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]

  SHAPES = {}
  SHAPES['root'] = polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY])
  SHAPES['hip'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  SHAPES['knee'] = polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2))
  SHAPES['arm'] = polygonShape(box=(ARM_W / 2, ARM_H / 2))
  SHAPES['claw'] = polygonShape(box=(CLAW_W / 2, CLAW_H / 2))

  clawMask = 0x011
  maskBits = 0x001
  bodies = {
      'lhip': Body(SHAPES['hip']),
      'lknee': Body(SHAPES['knee']),
      'rhip': Body(SHAPES['hip']),
      'rknee': Body(SHAPES['knee']),
      'shoulder': Body(SHAPES['arm'], maskBits=maskBits, density=0.1),
      'elbow': Body(SHAPES['arm'], maskBits=maskBits, density=0.1),
      'lclaw0': Body(SHAPES['claw'], maskBits=clawMask, density=0.1),
      'lclaw1': Body(SHAPES['claw'], maskBits=clawMask, density=0.1),
      'rclaw0': Body(SHAPES['claw'], maskBits=clawMask, density=0.1),
      'rclaw1': Body(SHAPES['claw'], maskBits=clawMask, density=0.1),
  }
  joints = {
      'lhip': Joint('root', 0.05, (0.0, LEG_DOWN), (0, LEG_H / 2), [-0.8, 1.1]),
      'lknee': Joint('lhip', 0.05, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.6, -0.1]),
      'rhip': Joint('root', -0.05, (0.0, LEG_DOWN), (0, LEG_H / 2), [-0.8, 1.1]),
      'rknee': Joint('rhip', -0.05, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.6, -0.1]),

      'shoulder': Joint('root', 2.0, (0, 5 / SCALE), (0, -ARM_H / 2), [-3.0, 3.0], limited=False),
      'elbow': Joint('shoulder', 3.0, (0, ARM_H / 2), (0, -ARM_H / 2), [-2.0, 2.0], limited=False),
      'lclaw0': Joint('elbow', 2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-2.0, 1.0]),
      'lclaw1': Joint('lclaw0', 3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
      'rclaw0': Joint('elbow', -2.25, (0, ARM_H / 2), (0, -CLAW_H / 2), [-1.0, 2.0]),
      'rclaw1': Joint('rclaw0', -3.75, (0, CLAW_H / 2), (0, -CLAW_H / 2), [0.0, 0.0]),
  }

  return Robot(
      type=robot.type,
      name=robot.name,
      root_body=Body(SHAPES['root']),
      bodies=bodies,
      joints=joints,
  )

@register('luxo')
def make_luxo(robot, C):
  VERT = 10 / SCALE
  SIDE = 5 / SCALE
  LEG_W, LEG_H = 8 / SCALE, 24 / SCALE
  LL_H = 20 / SCALE
  LUXO_POLY = A[(-15, +15), (+20, +25), (+20, -25), (-15, -15)] * 0.8
  SHAPES = {}
  SHAPES['root'] = polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LUXO_POLY])
  SHAPES['hip'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  SHAPES['knee'] = polygonShape(box=(0.8 * LEG_W / 2, LL_H / 2))
  SHAPES['foot'] = polygonShape(box=(LEG_H, LEG_W / 2))
  return Robot(
      type=robot.type,
      name=robot.name,
      root_body=Body(SHAPES['root'], density=0.1, maskBits=0x011),
      bodies={
          'lhip': Body(SHAPES['hip']),
          'lknee': Body(SHAPES['knee']),
          'lfoot': Body(SHAPES['foot']),
      },
      joints={
          'lhip': Joint('root', -0.5, (-SIDE, -VERT), (0, LEG_H / 2), [-0.1, 0.1]),
          'lknee': Joint('lhip', 0.5, (0, -LEG_H / 2), (0, LL_H / 2), [-0.9, 0.9]),
          'lfoot': Joint('lknee', 0.0, (0, -LEG_H / 2), (0, LEG_W / 2), [-0.5, 0.9]),
      },)


@register('gingy')
def make_gingy(robot, C):
  # TODO: make armless crab version.
  VERT = 10 / SCALE
  SIDE = 2 / SCALE
  #LEG_W, LEG_H = 8/SCALE, 16/SCALE
  BODY_W, BODY_H = 8 / SCALE, 25 / SCALE
  ARM_W, ARM_H = 8 / SCALE, 25 / SCALE
  LEG_W, LEG_H = 8 / SCALE, 30 / SCALE
  SHAPES = {}
  SHAPES['root'] = circleShape(radius=10 / SCALE)
  SHAPES['body'] = polygonShape(box=(BODY_W / 2, BODY_H / 2))
  SHAPES['arm'] = polygonShape(box=(ARM_W / 2, ARM_H / 2))
  SHAPES['leg'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  bodies = {
      'body': Body(SHAPES['body'], density=1.0),
      'larm': Body(SHAPES['arm'], maskBits=0x011),
      'rarm': Body(SHAPES['arm'], maskBits=0x011),
      'llarm': Body(SHAPES['arm'], maskBits=0x011),
      'rlarm': Body(SHAPES['arm'], maskBits=0x011),
      'lleg': Body(SHAPES['leg'], density=1.0),
      'rleg': Body(SHAPES['leg'], density=1.0),
  }
  joints = {
      # legs
      'body': Joint('root', 0.0, (0, -VERT), (0, BODY_H / 2), [-0.1, 0.1]),
      'larm': Joint('body', 1.5, (-SIDE, +VERT), (0, ARM_H / 2), [-1.5, 0.8]),
      'rarm': Joint('body', -1.5, (SIDE, +VERT), (0, ARM_H / 2), [-1.5, 0.8]),
      'llarm': Joint('larm', 1.5, (0, -ARM_H / 2), (0, ARM_H / 2), [-1.5, 1.5]),
      'rlarm': Joint('rarm', -1.5, (0, -ARM_H / 2), (0, ARM_H / 2), [-1.5, 1.5]),
      'lleg': Joint('body', 0.8, (-SIDE, -VERT), (0, LEG_H / 2), [-0.2, 0.4]),
      'rleg': Joint('body', -0.8, (SIDE, -VERT), (0, LEG_H / 2), [-0.4, 0.2]),
  }
  return Robot(type=robot.type, name=robot.name, root_body=Body(SHAPES['root'], density=0.01), bodies=bodies, joints=joints)

@register('octo')
def make_octo(robot, C):
  LEG_W, LEG_H = 8 / SCALE, 25 / SCALE
  SHAPES = {}
  SHAPES['root'] = circleShape(radius=1.5 * LEG_W)
  SHAPES['leg'] = polygonShape(box=(LEG_W / 2, LEG_H / 2))
  root_body = Body(SHAPES['root'], density=0.1)
  bodies = {
      'aleg1': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'bleg1': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'cleg1': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'dleg1': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'aleg2': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'bleg2': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'cleg2': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
      'dleg2': Body(SHAPES['leg'], maskBits=0x011, density=1.0),
  }
  joints = {
      'aleg1': Joint('root', 0.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'bleg1': Joint('root', 1.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'cleg1': Joint('root', 2.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'dleg1': Joint('root', 3.0, (0, 0), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'aleg2': Joint('aleg1', 0.0, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'bleg2': Joint('bleg1', 1.0, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'cleg2': Joint('cleg1', 2.0, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.0, 1.0], limited=False),
      'dleg2': Joint('dleg1', 3.0, (0, -LEG_H / 2), (0, LEG_H / 2), [-1.0, 1.0], limited=False),

  }
  return Robot(type=robot.type, name=robot.name, root_body=root_body, bodies=bodies, joints=joints, rand_angle=1)
