![](./assets/sideside.png)

boxLCD 📟
=================

boxLCD is box2D physics with low resolution and binarized rendering. It provides sample
environments and an API for defining worlds.

The aim of this project is to accelerate progress in [learned simulator](https://matwilso.github.io/robot-learning/learned-sims/) and world model research,
by providing a simple testbed for learning predictive dynamics models of physics environments.
Eventually we care about predictive models that are trained on real world data and that help robots act in the real world.
However, we believe these is a lot of fundamental research to do before we can realize that [full vision](https://matwilso.github.io/robot-learning/future/),
and that small scale testbeds are very useful for making progress.

boxLCD can be thought of as something akin to MNIST, but for learning dynamics models in robotics.
Generating MNIST digits is not very useful and has become fairly trivial.
But it provides a simple first task to try ideas on and it lets you iterate quickly and build intuition.
Learning dynamics models of 2D physics with low resolution images is not very useful and will be trivial
compared to learning models of the real world.
But it provides a much more tractable starting point, both for the field as a whole, as well as individuals starting out in the area.

boxLCD is somewhat of a minimum viable product at this point.
For more of the reasoning behind it and future plans, see the [Roadmap](#roadmap).

**Table of Contents**
- [Installation ‍💻](#installation-)
- [Environment demos ⚽](#environment-demos-)
- [Example training results 📈](#example-training-results-)
  - [Urchin](#urchin)
  - [Intelligent Domain Randomization](#intelligent-domain-randomization)
- [Roadmap 📍](#roadmap-)
  - [Future Features](#future-features)
- [Related Work 📚](#related-work-)

## Installation ‍💻

I recommend cloning the repo and experimenting with it locally, as you may want to read through things and customize them.

```
git clone https://github.com/matwilso/boxLCD.git
cd boxLCD
pip install -e .
pip install -r requirements.txt
```

## Environment demos ⚽

```python
from boxLCD import envs, C
env = envs.Dropbox(C) # for example
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
    env.render(mode='human')
```

Pretty rendering &#124; LCD rendering (upscaled) |  
:-------------------------:|
`envs.Dropbox()` (16x16) | 
![](./assets/demos/dropbox.gif)  |  
`envs.Bounce()` (16x16) | 
![](./assets/demos/bounce.gif)  |  
`envs.Urchin()` (16x16) | 
![](./assets/demos/urchin.gif)  |  
`envs.UrchinBall()` (16x24) | 
![](./assets/demos/urchin_ball.gif)  |  
`envs.UrchinBalls()` (16x32) | 
![](./assets/demos/urchin_balls.gif)  |  
`envs.UrchinCubes()` (16x32) | 
![](./assets/demos/urchin_cubes.gif)  |  


## Example training results 📈

See [examples](./examples) for scripts to recreate these.

| | Training Results |   |
|:---:|:-------------------------:| :-------------------------:|
|`envs.Dropbox`| after 10 epochs |  after 100 epochs |
|episode length: 100<br/># of parameters: 4.5e5<br/>training time: **3 minutes 25 seconds** |![](./assets/samples/dropbox-10.gif)  |  ![](./assets/samples/dropbox-100.gif) |
|`envs.Bounce()`| after 10 epochs | after 100 epochs |
|episode length: 200<br/># of parameters: 4.7e5<br/>training time: **6 minutes 29 seconds** |![](./assets/samples/bounce-10.gif)  |  ![](./assets/samples/bounce-100.gif) |
|`envs.Urchin()`| after 10 epochs | after 100 epochs |
|episode length: 200<br/># of paremeters: 2.5e6<br/>training time: **16 minutes 16 seconds** |![](./assets/samples/urchin-10.gif)  |  ![](./assets/samples/urchin-100.gif) |

To demonstrate what is possible with boxLCD, I trained a [model](./examples/model.py) on a few simple environments using a very naive approach.

It's a causally masked Transformer trained to predict the next frame given all past frames.
It is similar to a language model (e.g., GPT), but each token is simply the flattened 2D image for that timestep.
To train the model, we feed those flat image tokens in and the model produces independent Bernoulli distributions for 
each pixel, and we optimize this to match the ground truth. 
To sample the model, we prompt it with the start 10 frames of the episode, and have it predict
the rest autoregressively.

This is an extremely simplistic approach and it has to generate all pixels at once by sampling them independently.
In some ways, it's surprising it works.
For more details, see the code in [examples](./examples).

We do not condition on or try to predict continuous proprioceptive state, because I haven't gotten that working yet.
I find using Gaussians leads to very bad autoregressive samples.
Discrete sampling works much better out of the box.

### Urchin
The Urchin task is actually quite tricky and the model started to overfit the smallish dataset of 10k rollouts in this experiment.
The robot is 3-way symmetric, and since we are only using images here, the model is continually forced to
identify which leg corresponds to which index in the action vector based on past observations and actions.
We also randomly sample the actions for the 3 joints at each time step, so the agent can't rely on a semi-fixed policy
to narrow down the state space it has to cover.

### Intelligent Domain Randomization
Because powerful generative models will have to model uncertainty in the environment, sampling them 
will give you intelligent domain randomization for free. Instead of randomizing over a bunch of wacky parameters,
your model will be tuned to the underlying distribution and only give you variety you might actually see in the real world.

For a rough proof of concept of this, I created an environment that simulates either the falling
box or the circle. Since these shapes are sometimes indistinguishable at low resolution, the model 
cannot tell them apart given the prompt. If we were training a robot to manipulate these objects, we
would want it to be prepared for either scenario.

Below is a cherry picked example, where the desired behavior occurs.
On the far right 2 rollouts, the model is uncertain about the shape and happens to sample the wrong
one---a box instead of a circle, and a circle instead of a box.

![](./assets/samples/domrand_good.gif) 

It doesn't always do this, and sometimes it just waffles between bouncing and not.
But the model I used is extremely naive and doesn't allow sampling in a latent space,
so I expect more sophisticated models to do better.

## Roadmap 📍

Some of the reasoning behind this project can be found in some blog posts I have written on 
the [future of robot learning](https://matwilso.github.io/robot-learning/future/), and [learned simualtors](https://matwilso.github.io/robot-learning/learned-sims/).

boxLCD tries to capture some key properties of future learned simulators:
- **physics based.** unlike past related work, robots and objects don't move magically. they are governed by consistent physics and joints must be actuated to propel the robot.
- **vision-based.** you primarily sense the real world through vision (pixels).
- **partially observable.** even what you can currently see doesn't tell the full story of the world. you constantly have to make estimates of state that you only observe indirectly. making reasonable continuations of physics prompts that are plausible given all knowledge. and reasonable sampling over unknowns. and [intelligent domain randomization](#intelligent-domain-randomization).
- **interfaceable.** enable loading of structured information into predictions, like feeding meshes, natural language descriptions. 

While being computational tractable and easy to work with:
- **narrow 2d physics settings**, at least to start out.
- **simple rendering.** boxLCD enables variable sized rendering, but the default envs use a maximum `16x32 = 544` sized binary images (smaller than MNIST). compared to datasets like the [BAIR Pushing dataset](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small) with `64x64x3 = 12288` sized RGB images, this represents a 24x descrease in floating point numbers on the input and output. And information-wise, `24*8bits=`192x decrease in the bits to process, which can matter especially for naive PixelCNN type approaches.
- **programmatic and customizable.** you can geneate new scenarios and customize the environments to different settings you want to test.

boxLCD is in active development.
Right now, we are focused on developing environments and training models solely to predict accuracte physics, given past observations and actions.

### Future Features
- goal-based tasks and leverage our models to quickly learn to solve them.
  - maybe something like [block dude](https://www.calculatorti.com/ti-games/ti-83-plus-ti-84-plus/mirageos/block-dude/) but full physics based
- more robots and varied objects
- support for scrolling (environments which do not fit on the screen all at once)
- static environment features like ramps and walls
- maybe multiple image channels to represent these different layers 
- more formal benchmarks and bits/dim baselines

## Related Work 📚

https://github.com/kenjyoung/MinAtar

There are some related work, like moving MNIST. But that doesn't have control in it. Also this is lower dim.
https://www.tensorflow.org/datasets/catalog/moving_mnist

Doom and Berkeley dataset are other examples. But they are higher res and less configurable.
They also don't have associated structured information.

This is not like anything else exists. It is explicitly targeting the world models
and learned simulator task and provides low dimensional stuff. It allows custom environment interaction
to test custom scenarios. You incorporate robot actions and robot proprioception with partial observations,
as we will have in the real world.

- targeting learned simulator and world model research goals
- extremely low-res and binary for quick iteration speed
- video prediction, integrated with robot action *and* proprioception, as will be the case in the real world
- greater access to the simulator to enable custom scenarios, not just a fixed set of envs. but more general settings

This is explicitly building to the goal of learned sims and world models.

Background knowledge pointer to faq


https://haozhi.io/RPIN/

https://phyre.ai/
This one you set the state of the environment and then you see it roll out.
This is unlike robotics where you act at every timestep.
It is a narrow setting where you take one action and see what happens for many steps.

Also taking actions at every step is way harder to learn.
This creates many possible ways states can diverge. You can't rely
on them following a sequence, which is much easier.

Ilya bouncing balls. Back in the day though, these were not binarized, slightly larger.
And the RTRBM (Recurrent Temporal Restricted Boltzmann Machine) produces results that are not crisp.
The balls move, but the collisions are gooey (in supplementary, compare 1.gif,2.gif with the training data 5.gif).
https://papers.nips.cc/paper/2008/hash/9ad6aaed513b73148b7d49f70afcfb32-Abstract.html

Basically the point is this project is trying to approximate something very specific.
No previous work are trying to learn simulators or world models like this.
And the details and what you aim for matter. 
We think this aims most closely at an interesting goal
We think this has the greatest pareto front / product / AUC of aiming to the goal and being more approachable with small budgets.



https://github.com/greydanus/mnist1d
https://greydanus.github.io/2020/12/01/scaling-down/

This is not that similar, but has the shared goal of focusing on iteration speed.
We should focus on things that will scale. It may not help a ton for what actually ends
up scaling, but it gives a good place to build intuitions and fundamentals.

