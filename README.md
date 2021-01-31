![](./assets/sideside.gif)
# 📟 boxLCD 📟
## box2D physics with MNIST-sized rendering

boxLCD renders simple 2D physics environments in very low resolution binarized images to enable quick
iterations of ideas on low computational budgets.
It offers a few basic sample environments and a limited wrapper around the pybox2d API to make defining custom b2worlds easier.

The goal of this project is to accelerate progress in learned simulator and world model research,
ultimately so that robots can learn strong predictive models and act more effectively in the real world.

2D physics, with very-low resolution offers a crude approximation to the final task. But we think it captures
enough of the structure of the final goal to be useful, while enabling tremendously quicker iteration
cycles for trying out ideas, which will lower the bar to entry of working on these problems and accelerate progress.

// TODO: add docs on
// TODO: specific envs and stuff. like the options and the custom creatures you can create
// TODO: we have purely state-based and also mixture of state and pixel-based prediction
// TODO: make organization better

## boxLCD Challenge 🌐 🧩

Associated with boxLCD is the boxLCD Challenge.
Meant to simulate as closely as possible the challenges that will be faced in developing powerful learned world models
on real robots in the real world, while being tremendously more tractable.

We execute a robot with random actions in the world to generate as much data as we want. 
And then the goal is to learn a model that predicts the future with the lowest possible error.
By pushing on this metric, we will be able to develop better methods for learning models of the world
that will eventually be useful for enabling robots to learn to achieve more difficult tasks than is possible today.

The underlying physics and logic are quite complex, with contacts, momentum, and robot actions,
but the dimensionality shown in the gif above is only 16x32 = 512, compared to 64x128x3 = 24576. So 50x less data, and smaller than MNIST (28x28=784).
So the task is complex and meaningful, but experiments can be run with much less compute.
It is complex, mirrors challenges we will face in real robotics, and it should offer good signal about what really matters.

There are precise rules the govern it, just like the real world.
You can make observations about those rules.
And you want to learn how the system works.


Properties that will likely be useful in real world learned sims and world models:
- merging robot proprioception with exteroception (cameras, sensors)
- not relying on raw state knowledge of objects
- enable loading of structured information into predictions
- modeling uncertainty, making reasonable continuations of physics prompts that are plausible given all knowledge. and reasonable sampling over unknowns

// convert -resize 100% -delay 2 -loop 0 *.png test.gif

## FAQS
**Why is it called boxLCD?**
The rendering looks kind of like an LCD display. And it rhymes with 2d.