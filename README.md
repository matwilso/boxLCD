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

// TODO: create examples of bounce, dropbox, agent with varieties. w/ both pyglet and lcd rendering.

I think framing this, building this, and working on it will end up being valuable.


## Related Work

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



## Expected speed gains: some back of the envelope computations

### Let's talk computation

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

### Let's talk bits

The PixelCNN numbers are a bit unfair and more recent architecture (PixelCNN++ and later) use more clever approaches.
Also other approaches don't have to generate pixels one at a time and can greatly amortize the computations.
But information-wise, there are 8-bits to represent an RGB pixel. So that would make it 64x128x3x8 = 1.97 million, or 384x more information
to process in the case of the large resolution image.

Binary saves you 24x the amount of information as RGB and scaling down the image size accounts for the rest.

Basically, low-res binarized images are very small to process and the gains in iteration speed from these are pretty insane.
It can be easy to glaze over these numbers. But 50x iteration speed means you can run a test in 1 hr vs. 2 days.
That is the difference of being able to actually work on something and not.

### Let's talk caveats

Now this does assume that a large cost of computations is the I/O, where maybe the underlying
data process is the tricky part. Since both types of rendering represent the same underlying data
process, maybe your gains are weaker.
But when you balance all these factors out, I think this approach is likely to really help the speed
at which you can try ideas and see useful results and iterate.


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


Properties that will likely be useful in real world learned sims and world models:
- merging robot proprioception with exteroception (cameras, sensors)
- not relying on raw state knowledge of objects
- enable loading of structured information into predictions
- modeling uncertainty, making reasonable continuations of physics prompts that are plausible given all knowledge. and reasonable sampling over unknowns

// convert -resize 100% -delay 2 -loop 0 *.png test.gif

## FAQS
**Why is it called boxLCD?**
The rendering looks kind of like an LCD display. And it rhymes with 2d.