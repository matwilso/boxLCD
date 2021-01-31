![](./assets/sideside.gif)
# boxLCD 📟 - box2D physics with MNIST-style rendering

boxLCD enables rendering in very low resolution binarized images to enable quick
iterations of ideas on low computational budgets.
It offers a few basic sample environments and a limited wrapper around the pybox2d API to make defining custom b2worlds easier.

The goal of this project is to accelerate progress in learning world models and data-driven physics simulators (learned simulators).
By using very low dimensional rendering, researchers can work on developing pixel-based approaches that
can operate in the real world, while being able to iterate on ideas much more quickly.
The underlying physics and logic are quite complex, with contacts, momentum, and robot actions,
but the dimensionality shown in the gif above is only 16*32 = 512, compared to 64*128*3 = 24576. So 50x less data, and smaller than MNIST (28x28=784).

Associated with boxLCD is the **boxLCD Challenge** 🌐 🧩
Meant to simulate as closely as possible the challenges that will be faced in developing powerful learned world models
on real robots in the real world, while being tremendously more tractable.

We execute a robot with random actions in the world to generate as much data as we want. 
And then the goal is to learn a model that predicts the future with the lowest possible error.
By pushing on this metric, we will be able to develop better methods for learning models of the world
that will eventually be useful for enabling robots to learn to achieve more difficult tasks than is possible today.


It's called boxLCD because the rendering is reminscent of monochromatic LCD displays, and it rhymes.


I think this is a really good task because the data is really small but the process that generates the data is very complex.
So you can run experiments with much less compute. But it is challenging and it should offer good signal about what really matters.
Also, everything will be set up to simulate real world robot challenges well.

Properties that will likely be useful in real world learned sims and world models:
- merging robot proprioception with exteroception (cameras, sensors)
- not relying on raw state knowledge of objects
- enable loading of structured information into predictions
- modeling uncertainty, making reasonable continuations of physics prompts that are plausible given all knowledge. and reasonable sampling over unknowns

// convert -resize 100% -delay 2 -loop 0 *.png test.gif