examples
=================


## Basic

Just run an env and visualize the human mode rendering:
```
python basic.py
```

Similar to basic, but allows you to set the env from cmd line, and it uses some keyboard cmds (ESC - quit, SPACE - pause, RIGHT - step env, 0 - reset env)
```
python -m less_basic.py --env=UrchinBall
```

## Training

Simple naive training approach using a Tranformer, where LCD (~16x16) frames are flattened and used directly as
tokens. For training and sampling, entire frames are produced at a time and each individual pixel is sampled.

For simplicity, and because I haven't gotten it working well yet, we do not predict or deal with continuous proprioception state information,
just images

1. Collect data:
```
python collect.py --env=Bounce --collect_n=10000 # (10k rollouts, should take 5-ish minutes)
```
2. Train a model on that data:
```
python train.py --env=Bounce --datapath=rollouts/Bounce-10000.npz # trains in a few minutes on GPU, longer for harder tasks
```
3. Visualize training and samples:
```
tensorboard logs/
```