# SHELL COMMANDS TO RUN THE CODE

## Overview:

Step 1: collect data <br>
Step 2: learn models on that data <br>
Step 3: do RL. perhaps inside of a learned model <br>

- [Overview:](#overview)
- [Collect data](#collect-data)
- [Arbiter training](#arbiter-training)
- [MODEL LEARNING](#model-learning)
  - [Autoencoders](#autoencoders)
  - [Video](#video)
  - [Model Evaluations](#model-evaluations)
- [RL](#rl)
  - [BodyGoals in real environment](#bodygoals-in-real-environment)
  - [BodyGoals in learned environment](#bodygoals-in-learned-environment)
  - [CubeGoals in real environment](#cubegoals-in-real-environment)

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
python -m research.main --mode=train --model=MultiStepArbiter --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/Urchin --total_itr=30000  --nfilter=64 --hidden_size=256 --window=5
```

```bash
# all envs
python3 scripts/kicker.py arbiter --model=MultiStepArbiter
```

## MODEL LEARNING

### Autoencoders

// should run for about 70s per 1000 training iterations on my 1080Ti. <br>
// maybe about 10 mins or so total, until the error in the pstate stuff goes away. around -6 log mse

#### BVAE

```bash
# single env
python -m research.main --mode=train --model=BVAE --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/encoder/BVAE/Urchin --total_itr=30000 --total_itr=30000 --hidden_size=64 --vqK=64 --vqD=16 --nfilter=16 --window=5
```

```bash
# all envs
python3 scripts/kicker.py train --model=BVAE
```

#### RNLDA

```bash
# single env
python -m research.main --mode=train --model=RNLDA --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/encoder/RNLDA/Urchin --total_itr=30000 --total_itr=30000 --hidden_size=64 --vqK=64 --vqD=8 --nfilter=16 --window=5
```

```bash
# all envs
python3 scripts/kicker.py train --model=RNLDA
```

### Video

#### RSSM

```bash
# single env
python -m research.main --mode=train --model=RSSM --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/video/RSSM/Urchin --total_itr=100000 --total_itr=100000 --arbiterdir=logs/trash/Urchin --nfilter=64 --hidden_size=300 --free_nats=0.01
```

```bash
# all envs
python3 scripts/kicker.py train --model=RSSM
```

#### FIT (Flat Image Token)
```bash
# single env
python -m research.main --mode=train --model=FIT --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/video/FIT/Urchin --total_itr=100000 --total_itr=100000 --arbiterdir=logs/trash/Urchin --n_layer=2 --n_head=4 --n_embed=256 --hidden_size=256
```

```bash
# all envs
python3 scripts/kicker.py train --model=FIT
```

#### FBT (Flat Binary Token)
```bash
# single env
python -m research.main --mode=train --model=FBT --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/video/FBT/Urchin --total_itr=100000 --total_itr=100000 --arbiterdir=logs/trash/Urchin --n_layer=4 --n_head=8 --n_embed=512 --hidden_size=512 --weightdir=logs/trash/encoder/BVAE/Urchin
```

```bash
# all envs
python3 scripts/kicker.py train --model=FBT
```

#### FRNLD (Flat Ronald)
```bash
# single env
python -m research.main --mode=train --model=FRNLD --lr=0.0005 --bs=32 --log_n=1000 --datadir=logs/trash/Urchin --logdir=logs/trash/video/FRNLD/Urchin --total_itr=100000 --total_itr=100000 --arbiterdir=logs/trash/Urchin --n_layer=4 --n_head=8 --n_embed=512 --hidden_size=512 --weightdir=logs/trash/encoder/RNDLA/Urchin
```

```bash
# all envs
python3 scripts/kicker.py train --model=FRNLD
```

### Model Evaluations

```
python -m research.main --mode=eval --env=UrchinCube --datadir=logs/datadump/UrchinCube/ --arbiterdir=logs/arbiter/UrchinCube --model=FBT --prompt_n=3 --weightdir=logs/april28/video/FBT/UrchinCube/ --logdir=logs/evals/FBT_UrchinCube --bs=500
```

```
python -m research.main --mode=eval --env=Urchin --datadir=logs/datadump/Urchin/ --arbiterdir=logs/arbiter/Urchin --model=RSSM prompt_n=3 --weightdir=logs/video/RSSM/Urchin/ --logdir=logs/april22/eval/RSSM_Urchin --bs=1000
```


## RL

### BodyGoals in real environment
```bash
python rl/main.py ppo --env=Luxo --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --logdir=logs/rl/Luxo_real/ --total_steps=500000 --goal_thresh=0.05

python rl/main.py ppo --env=Urchin --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --logdir=logs/rl/Urchin_real/ --total_steps=1000000 --goal_thresh=0.05
```

### BodyGoals in learned environment
```bash
python rl/main.py ppo --env=Luxo --model=FBT --weightdir=logs/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --lenv=1 --logdir=logs/rl/Luxo_lenv --lenv_temp=1.0 --total_steps=500000 --goal_thres=0.05

python rl/main.py ppo --env=Urchin --model=FBT --weightdir=logs/video/FBT/Urchin/ --window=50 --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --lenv=1 --logdir=logs/rl/Urchin_lenv --lenv_temp=1.0 --total_steps=1000000 --goal_thres=0.05
```


### CubeGoals in real environment
```
python rl/main.py ppo --env=UrchinCube --goals=1 --num_envs=24 --bs=4096 --hidden_size=256 --goal_thres=0.05 --pi_lr=1e-4 --vf_lr=1e-4 --state_key=full_state --diff_delt=1
```

<!--

env=LuxoCube
DP=logs/datadump/10fps/luxocube/
python main.py --mode=train --env=$env --datadir=$DP --model=bvae --window=4 --bs=64 --log_n=1000   --lr=1e-3 --skip_train=0 --vqK=64 --hidden_size=64 --nfilter=64 --vqD=32 --log_n=100 --logdir=logs/bvae/x
python main.py --mode=train --env=$env --datadir=$DP --model=flatb --window=100 --bs=32 --log_n=1000 --lr=1e-3 --weightdir=$WD --n_layer=3 --n_head=8 --hidden_size=512 --n_embed=512 --log_n=100 --logdir=logs/flatb/luxocube/bigger

BVAE preproc, real env
python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatb --weightdir=logs/flatb/bigger/ --window=100 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/rl/flatb/nolenv/bvae_preproc_bs64_fixgoal --lenv_temp=0.1 --bs=64 --hidden_size=128 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0 --lenv=0 --net=bvae




```
# test learned simulator

python learned_env.py --env=Luxo --datadir=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --weightdir=logs/flatev/x/monsta2/ --goals=1 --num_envs=8 --window=100 

# run RL on learned simulator env, and non-learned env

python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/x/ --window=100 --goals=1 --num_envs=8 --lenv=1 --logdir=logs/rl/lenv/x --lenv_temp=0.1 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0
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
-->