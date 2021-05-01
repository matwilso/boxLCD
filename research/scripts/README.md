# COMMON SHELL COMMANDS THAT I SAVE SO I DON'T FORGET THEM

## Overview:

Step 1: collect data <br>
Step 2: learn models on that data <br>
Step 3: do RL. perhaps inside of a learned model <br>

- [Collect data](#collect-data)
- [MODEL LEARNING](#model-learning)
  - [Autoencoding](#autoencoding)
  - [VideoModels](#videomodels)
- [RL](#rl)
  - [CUBES](#cubes)
- [MISC](#misc)

## Collect data

```bash
# single env
python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env=Urchin --logdir=logs/trash/Urchin
```

```bash
# all envs
python3 scripts/kicker.py collect
```

## Arbiter training

```bash
# single env
python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env=Urchin --logdir=logs/trash/Urchin
```

```bash
# all envs
python3 scripts/kicker.py collect
```



```
python main.py --mode=train --env=Dropbox --datadir=logs/datadump/Dropbox/ --model=MultiStepArbiter --lr=5e-4 --log_n=1000 --bs=32 --nfilter=64 --hidden_size=256 --logdir=logs/evals/arbiter/Dropbox/ --window=5

```


## MODEL LEARNING

### Autoencoding 

```
args="--window=4 --bs=256"
```

#### Arbiter
```bash
python main.py --mode=train --env=Luxo --datadir=logs/datadump/Luxo/ --model=ArbiterAE --nfilter=32 --lr=1e-3 --logdir=logs/autoencoder/Luxo/ArbiterAE/better_enc_norelu_compile --log_n=1000 --bs=32 --window=4
```

#### VAE 

```bash
model=VAE
python main.py --mode=train --env=$env --datadir=logs/datadump/$env/ --model=$model  --window=4 --bs=256 --lr=1e-3  --logdir=logs/autoencoder/$env/$model/small 
```

#### Binary VAE 

```bash
model=BVAE
python main.py --mode=train --env=$env --datadir=logs/datadump/$env/ --model=$model  --nfilter=16 --vqD=8 --vqK=32 --hidden_size=64  --lr=1e-3  --logdir=logs/autoencoder/$env/$model/small $args
```


```bash
python main.py --mode=train --env=Luxo --datadir=logs/datadump/Luxo/ --model=BVAE --window=4 --log_n=1000 --lr=1e-3 --bs=64 --log_n=1000 --lr=1e-3 --vqK=64 --hidden_size=64 --logdir=logs/newfolder/autoencoder/bvae/x --lr=1e-3 --skip_train=0 --nfilter=64 --vqD=32 --log_n=1000 --arbiterdir=logs/newfolder/arbiter/singlestep/2/

# should run for about 70s per 1000 training iterations on my 1080Ti.
# maybe about 10 mins or so total, until the error in the pstate stuff goes away. around -6 log mse


```



env=LuxoCube
DP=logs/datadump/10fps/luxocube/
python main.py --mode=train --env=$env --datadir=$DP --model=bvae --window=4 --bs=64 --log_n=1000   --lr=1e-3 --skip_train=0 --vqK=64 --hidden_size=64 --nfilter=64 --vqD=32 --log_n=100 --logdir=logs/bvae/x
python main.py --mode=train --env=$env --datadir=$DP --model=flatb --window=100 --bs=32 --log_n=1000 --lr=1e-3 --weightdir=$WD --n_layer=3 --n_head=8 --hidden_size=512 --n_embed=512 --log_n=100 --logdir=logs/flatb/luxocube/bigger

BVAE preproc, real env
python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatb --weightdir=logs/flatb/bigger/ --window=100 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/rl/flatb/nolenv/bvae_preproc_bs64_fixgoal --lenv_temp=0.1 --bs=64 --hidden_size=128 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0 --lenv=0 --net=bvae


### VideoModels 

```
args="--window=4 --bs=256"
```

#### Flat Binary Token
```bash

model=FlatBToken
wd=logs/autoencoder/$env/BVAE/small
args="--window=100 --bs=100"

python main.py --mode=train --env=$env --datadir=logs/datadump/$env/ --model=$model --weightdir=$wd --n_layer=3 --n_head=8 --hidden_size=512 --n_embed=512 --logdir=logs/video/$model/x $args
```

#### Flat Everything
#### Flat Image



## LEARNED SIMULATION
```
# test learned simulator

python learned_env.py --env=Luxo --datadir=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --weightdir=logs/flatev/x/monsta2/ --goals=1 --num_envs=8 --window=100 

# run RL on learned simulator env, and non-learned env

python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/x/ --window=100 --goals=1 --num_envs=8 --lenv=1 --logdir=logs/rl/lenv/x --lenv_temp=0.1 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0
```

## RL

```bash
python rl/sac.py --env=Luxo --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=mlp --logdir=logs/rl/luxo
python rl/sac.py --env=Urchin --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=mlp --logdir=logs/rl/urchin
python rl/sac.py --env=Luxo --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=cnn --logdir=logs/rl/luxo_cnn
python rl/sac.py --env=Urchin --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=cnn --logdir=logs/rl/urchin_cnn
```

### CUBES

```bash
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0
python rl/sac.py --env=CrabCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0

#these 2 worked
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_1.5_2 --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=1 --fps=10
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/nodiffdelt_1.5 --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=0 --fps=10

#same
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_2.0_objchanges_halfmass --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0 --diff_delt=1 --fps=10
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_1.5_objchanges_halfmass --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=1 --fps=10

```

multi-step model
python main.py --mode=train --env=UrchinBall --datadir=$DP --model=multistep --vidstack=4 --phase=1 --log_n=50 --logdir=logs/biphase/x2/
python main.py --mode=train --env=UrchinBall --datadir=$DP --model=multistep --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/1e5/nl3_512_16_32stacks/ --n_layer=3 --n_embed=512 --n_head=16 --bs=16 --amp=1

frame_token
python main.py --mode=train --env=Luxo --datadir=$DP --model=frame_token --logdir=logs/luxo/flattie/ --lr=1e-3 --n_layer=3 --n_embed=512 --n_head=16 --lr=5e-4

FLAT EVERYTHING (GOOD ONE)
state vqvae so that state is discrete binary
python main.py --mode=train --env=Luxo --datadir=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=statevq --window=16 --log_n=1000 --lr=1e-3 --logdir=logs/ternary/juststate128_512_save/ --bs=32 --log_n=1000 --lr=1e-3 --vqK=128 --hidden_size=512
then flat everything model
 a
python main.py --mode=train --env=Luxo --datadir=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --logdir=logs/flatev/med_bs32_ESR --bs=32 --log_n=1000 --lr=1e-3 --weightdir=logs/ternary/juststate128_512_save/ --window=100 --n_layer=3 --n_head=16 --hidden_size=512 --n_embed=512
 b
python main.py --mode=train --env=Luxo --datadir=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --logdir=logs/flatev/100window/smallnet_bs32_8e-4_fix --bs=32 --log_n=1000 --lr=1e-3 --weightdir=logs/ternary/juststate128_512_save/ --window=100 --n_layer=3 --n_head=16 --hidden_size=512 --n_embed=512 --lr=8e-4

## MISC

```
# convert a set of images to a single video gif
convert -resize 100% -delay 2 -loop 0 *.png test.gif
```