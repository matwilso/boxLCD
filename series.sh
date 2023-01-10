
# define common args
COMMON="python /home/matwilso/code/boxLCD/research/main.py --mode=train --model=video_rin --datadir=logs/datadump/Bounce2_large/ --window=16 --dst_resolution=16 --save_n=1 --log_n=1000 --timesteps 100 --data_workers 4 --total_time 7200 --n_head 8"

SHORT_TIME="--total_time 360"
# 2hrs a pop
NORMAL_TIME="--total_time 7200"

# define extra experiments using the common args

# shallow 

# head stuff
#$COMMON $SHORT_TIME --bs 50 --logdir logs/jan08/vid/exp/head4 --lr 1e-4 --num_layers 1 --num_blocks 1 --n_z 128 --dim_z 512 --self_cond 0.5 --n_head 4
#$COMMON $SHORT_TIME --bs 50 --logdir logs/jan08/vid/exp/head8 --lr 1e-4 --num_layers 1 --num_blocks 1 --n_z 128 --dim_z 512 --self_cond 0.5 --n_head 8
#$COMMON $SHORT_TIME --bs 50 --logdir logs/jan08/vid/exp/head16 --lr 1e-4 --num_layers 1 --num_blocks 1 --n_z 128 --dim_z 512 --self_cond 0.5 --n_head 16


# general experiments. 
# we'll do coarse 2hr ones and then maybe go deeper on some. or should we do 1 hr ones.
# nah let's go 2. lean on the side of cautiously too long experiments.
$COMMON --bs 50 --logdir logs/jan08/vid/exp2/1 --lr 1e-4 --num_layers 2 --num_blocks 2 --n_z 128 --dim_z 512 --self_cond 0.5

$COMMON --bs 100 --logdir logs/jan08/vid/exp2/1 --lr 1e-4 --num_layers 2 --num_blocks 2 --n_z 128 --dim_z 512 --self_cond 0.5

$COMMON --bs 100 --logdir logs/jan08/vid/exp2/3 --lr 1e-4 --num_layers 2 --num_blocks 2 --n_z 128 --dim_z 512 --self_cond 0.9

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/4 --lr 1e-4 --num_layers 4 --num_blocks 6 --n_z 128 --dim_z 1024 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/5 --lr 1e-4 --num_layers 3 --num_blocks 4 --n_z 128 --dim_z 800 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/6 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 128 --dim_z 512 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/7 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 512 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/8 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 256 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/9 --lr 1e-4 --num_layers 2 --num_blocks 2 --n_z 32 --dim_z 1024 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/10 --lr 5e-4 --num_layers 2 --num_blocks 3 --n_z 128 --dim_z 512 --self_cond 0.5

$COMMON --bs 50 --logdir logs/jan08/vid/exp2/11 --lr 1e-5 --num_layers 4 --num_blocks 6 --n_z 128 --dim_z 1024 --self_cond 0.5