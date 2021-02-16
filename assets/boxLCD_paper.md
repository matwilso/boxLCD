# boxLCD Whitepaper

**Abstract (identical to main README.md)**

The aim of this project is to accelerate progress in [learned simulator and world model research](https://matwilso.github.io/robot-learning/future/),
by providing a simple testbed for developing and quickly iterating on predictive modeling ideas.

// TODO: table of contents

## Related Work 📝


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


## Expected speed gains: some back of the envelope computations 🖩

### Bits 📲

Information wise. 

The PixelCNN numbers are a bit unfair and more recent architecture (PixelCNN++ and later) use more clever approaches.
Also other approaches don't have to generate pixels one at a time and can greatly amortize the computations.
But information-wise, there are 8-bits to represent an RGB pixel. So that would make it 64x128x3x8 = 1.97 million, or 384x more information
to process in the case of the large resolution image.

Binary saves you 24x the amount of information as RGB and scaling down the image size accounts for the rest.

Basically, low-res binarized images are very small to process and the gains in iteration speed from these are pretty insane.
It can be easy to glaze over these numbers. But 50x iteration speed means you can run a test in 1 hr vs. 2 days.
That is the difference of being able to actually work on something and not.


### Computation 🧑‍💻

Computation wise, the advantages are still large but not as drastic.

In a 64x128x3 image represented in floating point, there are 24576 values that must be processed in your network,
and some multiple more than that to produce outputs.
Now say the best architecture for this task ends up being PixelCNN (not ++ version).
In the PixelCNN network (not ++ version), you decode each pixel into a 256-sized softmax, so this
requires another factor of 256 computations on the output for a grand total of 6.3 million on the output.
24576 values on the input and a maximum of 6.3 million on the output.

Now if we instead use a 16x32x1x1 using the same method, there are a grand total of 512 values that must be processed
in your network, and for a binary cross entropy loss, the number remains the same.
We save a factor of 50x on the input and a factor of 12,000x on the output.

// TODO: we should probably lean a bit heavier on bits and less on computation.


### Caveats

Now this does assume that a large cost of computations is the I/O, where maybe the underlying
data process is the tricky part. Since both types of rendering represent the same underlying data
process, maybe your gains are weaker.
But when you balance all these factors out, I think this approach is likely to really help the speed
at which you can try ideas and see useful results and iterate.



# Assumption behind

- iteration speed matters
- simplified 2d will be enough to work on useful ideas



# misc unprocessed

This enables trying generative modeling techniques that operate in pixel space, and
trying ideas that would normally be slow to iterate on or prohibitively expensive. 

In the best case, we get useful research that enables future progress on real world real robot tasks.
In the worst case, it is useful for me to quickly try out ideas and get up to speed on learning world models.



that enables us to develop useful approaches, but with much quicker iteration speed and smaller computational budgets.

boxLCD renders simple 2D physics environments in very low resolution binarized images to enable quick
iterations of ideas on low computational budgets.


boxLCD is a testbed for which usings simple bo2d physics rendering at extremely low-resolution.


To this end, boxLCD renders box2d physics in extremely low-resolution images (16x32, smaller than MNIST-sized images). As shown in the gif, this is a simple but rich space for displaying physical motion.





prototyping and demonstrating hte ideas that will b euseful in the real case.
so that we can work out ideas and so that we can prove the potential value of future reasearch.







 




This is in alpha stages, and I don't really expect to have a ton of users so I will probably
be improving and breaking things a lot and redesigning. But if you are using it, let me know and I 
can try not to mess up your stuff. And also improve stuff.




I think the value of this is it can capture enough of the problem to be useful
while being simple enough to enable quick iteration.
It seems like a decent place to get started, where if we can't solve the problem
really well for these tasks, I don't think we can expect to solve them for more complex systems.

By itself, I don't think this is the most interesting problem.
But if we can't master this, I don't think we have a chance at tackling harder domains.
I think the analogy with MNIST is good.
This is just for quickly iterating on ideas so that you can deploy them on tasks you actually care about.

I do think this problem is important for robotics.


The generative modeling task and learning the simulator seem extremely rich and challenging,
and contain a lot of the meat of the problem.

I suspect this will have more action bechmarks later. But I think this is somewhat
already covered. And that generative modeling of this world itself is likely to lead
to ideas useful for MBRL and other work.

It helps to focus on subsets of the full challenge. And I think world modeling itself
is a very large and challenging subset of the full challenge.


Specifically we are focused on predictive learning / generative modeling. Given a set of environments
with similar underlying physics, can we learn a good predictive model of this data?
Can we effectively learn to clone the physics simulator using neural network models?

For now, we are focusing on how well you can learn to predict this data.

We expect that learning large predictive models of the world is going to be critical
to the future success of robot learning.

Operating on video is extremely expensive and can make research progress and iteration
speed slow, especially for those with limited budgets.


2D physics, with very-low resolution offers a crude approximation to the final task. But we think it captures
enough of the structure of the final goal to be useful, while enabling tremendously quicker iteration
cycles for trying out ideas, which will lower the bar to entry of working on these problems and accelerate progress.


The images are actually generally smaller than MNIST, but it is much more interesting.



## boxLCD Challenge 🌐 🧩

Associated with boxLCD is the boxLCD Challenge.
Meant to simulate as closely as possible the challenges that will be faced in developing powerful learned world models
on real robots in the real world, while being tremendously more tractable.

We execute a robot with random actions in the world to generate as much data as we want. 
And then the goal is to learn a model that predicts the future with the lowest possible error.
By pushing on this metric, we will be able to develop better methods for learning models of the world
that will eventually be useful for enabling robots to learn to achieve more difficult tasks than is possible today.

The underlying physics and logic are quite complex, with contacts, momentum, and robot actions, but the dimensionality is tiny.


shown in the gif above is only 16x32 = 512, compared to 64x128x3x256 = 24576x256 = 6291456. So 50x less data or actually 12800x less data, and smaller than MNIST (28x28=784). Now we're talking. We should talk bits. And bits-wise, there is so much less information to process. About 10kx less.
So the task is complex and meaningful, but experiments can be run with much less compute.
It is complex, mirrors challenges we will face in real robotics, and it should offer good signal about what really matters.

There are precise rules the govern it, just like the real world.
You can make observations about those rules.
And you want to learn how the system works.
