
# define common args
COMMON="python /home/matwilso/code/boxLCD/research/main.py --mode=train --model=video_rin --datadir=logs/datadump/Bounce2_large/ --window=16 --dst_resolution=16 --save_n=1 --log_n=1000 --timesteps 100 --data_workers 4 --n_head 8"

SHORT_TIME="--total_time 360"
# 2hrs a pop
NORMAL_TIME="--total_time 3600"

#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/8 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 512 --self_cond 0.5
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/7 --lr 5e-4 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 512 --self_cond 0.5
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/9 --lr 5e-5 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 512 --self_cond 0.5
#
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/10 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5
#
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/11 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 1024 --self_cond 0.5
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/12 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 16 --dim_z 1024 --self_cond 0.5
#
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/13 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 16 --dim_z 512 --self_cond 0.5
#
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/14 --lr 1e-4 --num_layers 3 --num_blocks 4 --n_z 64 --dim_z 512 --self_cond 0.5
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/15 --lr 1e-4 --num_layers 4 --num_blocks 6 --n_z 64 --dim_z 512 --self_cond 0.5
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/16 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 64 --dim_z 512 --self_cond 0.9
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/18 --lr 1e-4 --num_layers 2 --num_blocks 1 --n_z 64 --dim_z 512 --self_cond 0.9
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/19 --lr 1e-4 --num_layers 1 --num_blocks 1 --n_z 64 --dim_z 512 --self_cond 0.9
#$COMMON --bs 50 --logdir logs/jan08/vid/exp3/20 --lr 1e-4 --num_layers 1 --num_blocks 2 --n_z 64 --dim_z 512 --self_cond 0.9

$COMMON --bs 50 --logdir logs/jan08/vid/exp3/17 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5