examples
=================


## Basic

Just run an env and visualize the human mode rendering 
`python basic.py`

## Training

Simple naive training approach using a Tranformer, where LCD (~16x16) frames are flattened and used directly as
tokens. For training and sampling, entire frames are produced at a time and each individual pixel is sampled.

For simplicity, and because I haven't gotten it working well yet, we do not predict or deal with continuous proprioception state information,
just images

1. Collect data: `python collect.py --env=bounce --collect_n=10000 # (10k rollouts, should take 5-ish minutes)`  
2. Train a model on that data: `python train.py --env=bounce --datapath=rollouts/bounce-10000.npz # training takes about 10-15mins on GPU`
3. Visualize training and samples: `tensorboard logs/`